"""
Trajectory GRPO Logger - Save generation details to disk turn-by-turn

Logs each turn immediately (while image is in memory) to avoid RAM buildup.
Saves:
- Images (saved turn-by-turn)
- Prompts (decoded text)
- Responses (model outputs)
- Rewards (saved at trajectory finalization)
- Metadata (step, trajectory_id, etc.)

Usage in GRPOTrainer:
    # During turn generation (line ~1567)
    logger.log_turn(step, prompt_idx, gen_idx, turn_idx, prompt_ids, completion, image, processing_class)

    # After all trajectories complete (line ~1665)
    logger.finalize_trajectories(trajectories, step, mode)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict


class TrajectoryLogger:
    """Save trajectory generation details to disk turn-by-turn (memory efficient)."""

    def __init__(
        self,
        output_dir: str,
        save_images: bool = True,
        save_prompts: bool = True,
        save_responses: bool = True,
        max_trajectories_per_step: Optional[int] = None,
    ):
        """
        Initialize trajectory logger.

        Args:
            output_dir: Directory to save trajectory logs
            save_images: Whether to save images
            save_prompts: Whether to save prompt text
            save_responses: Whether to save model responses
            max_trajectories_per_step: Max trajectories to save per step (None = all)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_images = save_images
        self.save_prompts = save_prompts
        self.save_responses = save_responses
        self.max_trajectories_per_step = max_trajectories_per_step

        # Create subdirectories
        if save_images:
            (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "trajectories").mkdir(exist_ok=True)

        # Track trajectories being logged
        # Key: (step, prompt_idx, gen_idx), Value: {turns: [...], metadata: {...}}
        self.active_trajectories = {}
        self.trajectories_logged_per_step = defaultdict(int)

    def log_turn(
        self,
        step: int,
        prompt_idx: int,
        gen_idx: int,
        turn_idx: int,
        prompt_ids: List[int],
        completion: str,
        image: Optional[Any] = None,
        processing_class: Optional[Any] = None,
    ):
        """
        Log a single turn immediately (memory efficient).

        Args:
            step: Global training step
            prompt_idx: Prompt index in batch
            gen_idx: Generation index (0 to num_generations-1)
            turn_idx: Turn index in trajectory
            prompt_ids: Token IDs of prompt
            completion: Model response text
            image: PIL Image (if present)
            processing_class: Tokenizer for decoding
        """
        # Check limit
        if (
            self.max_trajectories_per_step is not None
            and self.trajectories_logged_per_step[step] >= self.max_trajectories_per_step
        ):
            return

        traj_key = (step, prompt_idx, gen_idx)
        traj_id = f"step{step:06d}_p{prompt_idx:03d}_g{gen_idx:02d}"

        # Initialize trajectory if first turn
        if traj_key not in self.active_trajectories:
            self.active_trajectories[traj_key] = {
                "trajectory_id": traj_id,
                "turns": [],
                "metadata": {
                    "step": step,
                    "prompt_idx": prompt_idx,
                    "gen_idx": gen_idx,
                },
            }
            self.trajectories_logged_per_step[step] += 1

        turn_data = {
            "turn_idx": turn_idx,
        }

        # Save prompt text
        if self.save_prompts and processing_class is not None:
            prompt_text = processing_class.decode(prompt_ids, skip_special_tokens=True)
            turn_data["prompt"] = prompt_text

        # Save response
        if self.save_responses:
            turn_data["response"] = completion

        # Save image immediately
        if self.save_images and image is not None:
            image_filename = f"{traj_id}_turn{turn_idx:02d}.png"
            image_path = self.output_dir / "images" / image_filename
            try:
                # Check if it's a PIL Image
                if hasattr(image, 'save'):
                    image.save(image_path)
                    # Use relative path for portability
                    turn_data["image_path"] = f"images/{image_filename}"
                else:
                    turn_data["image_error"] = f"Not a PIL Image: {type(image)}"
            except Exception as e:
                turn_data["image_error"] = str(e)

        # Add turn to trajectory
        self.active_trajectories[traj_key]["turns"].append(turn_data)

    def finalize_trajectories(
        self,
        trajectories: List[List[Dict[str, Any]]],
        step: int,
        mode: str = "train",
    ):
        """
        Finalize trajectories by adding rewards and saving JSON files.

        Args:
            trajectories: Shape (num_prompts, num_generations)
            step: Global training step
            mode: "train" or "eval"
        """
        num_prompts = len(trajectories)
        num_generations = len(trajectories[0]) if num_prompts > 0 else 0

        for prompt_idx in range(num_prompts):
            for gen_idx in range(num_generations):
                traj_key = (step, prompt_idx, gen_idx)

                # Skip if not logged (exceeded limit)
                if traj_key not in self.active_trajectories:
                    continue

                traj = trajectories[prompt_idx][gen_idx]
                logged_traj = self.active_trajectories[traj_key]

                # Add final metadata
                logged_traj["metadata"].update({
                    "mode": mode,
                    "timestamp": datetime.now().isoformat(),
                    "trajectory_length": traj.get("trajectory_length", 0),
                    "done": traj.get("done", False),
                    "terminated_naturally": traj.get("terminated_naturally", False),
                })

                # Add reward
                logged_traj["reward"] = traj.get("reward", 0.0)

                # Save to JSON
                traj_id = logged_traj["trajectory_id"]
                json_path = self.output_dir / "trajectories" / f"{traj_id}.json"
                with open(json_path, "w") as f:
                    json.dump(logged_traj, f, indent=2)

                # Remove from active tracking
                del self.active_trajectories[traj_key]

        # Clean up any remaining trajectories (shouldn't happen, but safety)
        if self.active_trajectories:
            remaining = len(self.active_trajectories)
            self.active_trajectories.clear()
            print(f"Warning: {remaining} trajectories were not finalized properly")

    def generate_html_report(self, step: Optional[int] = None):
        """
        Generate HTML report for visualizing trajectories.

        Args:
            step: If specified, only generate report for this step
        """
        import glob

        # Find all trajectory files
        if step is not None:
            pattern = str(self.output_dir / "trajectories" / f"step{step:06d}_*.json")
        else:
            pattern = str(self.output_dir / "trajectories" / "*.json")

        trajectory_files = sorted(glob.glob(pattern))

        if not trajectory_files:
            print(f"No trajectories found for step {step}")
            return

        # Generate HTML
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Trajectory GRPO Logs</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .trajectory { background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metadata { background: #e3f2fd; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
        .turn { border-left: 3px solid #2196f3; padding-left: 15px; margin: 15px 0; }
        .turn-header { font-weight: bold; color: #1976d2; margin-bottom: 8px; }
        .prompt { background: #f5f5f5; padding: 10px; border-radius: 4px; margin: 8px 0; white-space: pre-wrap; }
        .response { background: #e8f5e9; padding: 10px; border-radius: 4px; margin: 8px 0; white-space: pre-wrap; }
        .image { max-width: 512px; border-radius: 4px; margin: 10px 0; }
        h1 { color: #1976d2; }
        .positive { color: #2e7d32; }
        .negative { color: #d32f2f; }
    </style>
</head>
<body>
    <h1>Trajectory GRPO Generation Logs</h1>
"""

        for traj_file in trajectory_files:
            with open(traj_file, "r") as f:
                traj_data = json.load(f)

            metadata = traj_data["metadata"]
            reward = traj_data.get("reward", 0.0)
            reward_class = "positive" if reward >= 0 else "negative"

            html += f"""
    <div class="trajectory">
        <div class="metadata">
            <strong>Trajectory ID:</strong> {traj_data['trajectory_id']} |
            <strong>Step:</strong> {metadata['step']} |
            <strong>Length:</strong> {metadata['trajectory_length']} turns |
            <strong>Reward:</strong> <span class="{reward_class}">{reward:.2f}</span>
        </div>
"""

            for turn in traj_data.get("turns", []):
                turn_idx = turn["turn_idx"]
                html += f"""
        <div class="turn">
            <div class="turn-header">Turn {turn_idx + 1}</div>
"""

                # Image
                if "image_path" in turn:
                    html += f'            <img src="{turn["image_path"]}" class="image" />\n'
                elif "image_error" in turn:
                    html += f'            <p style="color: red;">Image error: {turn["image_error"]}</p>\n'

                # Prompt
                if "prompt" in turn:
                    html += f"""
            <div class="prompt">
                <strong>Prompt:</strong><br>
                {turn["prompt"]}
            </div>
"""

                # Response
                if "response" in turn:
                    html += f"""
            <div class="response">
                <strong>Response:</strong><br>
                {turn["response"]}
            </div>
"""

                html += """
        </div>
"""

            html += """
    </div>
"""

        html += """
</body>
</html>
"""

        # Save HTML
        if step is not None:
            html_path = self.output_dir / f"report_step{step:06d}.html"
        else:
            html_path = self.output_dir / "report_all.html"

        with open(html_path, "w") as f:
            f.write(html)

        print(f"✓ HTML report saved to: {html_path}")
        return html_path
