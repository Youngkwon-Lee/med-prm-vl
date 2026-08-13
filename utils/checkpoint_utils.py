"""
Checkpoint utilities for Med-PRM
================================
Manages checkpointing for long-running scoring jobs.

Enables resumption after interruptions without re-processing completed items.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


class CheckpointManager:
    """
    Manages checkpoints for scoring pipeline.

    Features:
      - Save intermediate results periodically
      - Resume from checkpoint with --resume flag
      - Automatic backup of last 2 checkpoints
      - Atomic writes to prevent corruption
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_freq: int = 100,
        keep_last_n: int = 2
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            checkpoint_freq: Save checkpoint every N items
            keep_last_n: Number of old checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_freq = checkpoint_freq
        self.keep_last_n = keep_last_n

        self.current_checkpoint = None
        self.processed_count = 0

    def save_checkpoint(
        self,
        data: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        is_final: bool = False
    ) -> str:
        """
        Save checkpoint to disk.

        Args:
            data: Processed data so far
            metadata: Additional metadata (model path, parameters, etc.)
            is_final: Whether this is the final checkpoint

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().isoformat()
        suffix = "_final" if is_final else f"_{self.processed_count:06d}"
        checkpoint_name = f"checkpoint{suffix}.json"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint_data = {
            "timestamp": timestamp,
            "processed_count": self.processed_count,
            "is_final": is_final,
            "metadata": metadata or {},
            "data": data,
        }

        # Atomic write: write to temp file, then rename
        temp_path = checkpoint_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=4, ensure_ascii=False)

            # Atomic rename (works on Unix; on Windows, might fail if dest exists)
            temp_path.replace(checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path.name} ({len(data)} items)")

        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        self.current_checkpoint = str(checkpoint_path)

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[List[Dict], Dict]:
        """
        Load checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Tuple of (data, metadata)

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            json.JSONDecodeError: If checkpoint is corrupted
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            data = checkpoint_data.get("data", [])
            metadata = checkpoint_data.get("metadata", {})
            self.processed_count = checkpoint_data.get("processed_count", 0)

            print(f"✅ Checkpoint loaded: {checkpoint_path.name}")
            print(f"   Processed items: {self.processed_count}")
            print(f"   Data items: {len(data)}")

            return data, metadata

        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Corrupted checkpoint {checkpoint_path}: {e}",
                e.doc,
                e.pos
            )

    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Find latest non-final checkpoint in directory.

        Returns:
            Path to latest checkpoint or None
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))

        if not checkpoint_files:
            return None

        # Sort by modification time
        latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        return str(latest)

    def update_progress(self, count: int) -> Optional[str]:
        """
        Update progress counter. Save checkpoint if threshold reached.

        Args:
            count: Number of items processed so far

        Returns:
            Path to checkpoint if saved, None otherwise
        """
        self.processed_count = count

        if count > 0 and count % self.checkpoint_freq == 0:
            return self.current_checkpoint

        return None

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Keep final checkpoint and last N others
        keep = [f for f in checkpoint_files if "_final" in f.name][:1]  # Keep final
        keep += checkpoint_files[: self.keep_last_n]  # Keep last N

        for checkpoint_file in checkpoint_files:
            if checkpoint_file not in keep:
                try:
                    checkpoint_file.unlink()
                except Exception as e:
                    print(f"⚠️  Failed to delete {checkpoint_file}: {e}")

    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """
        Get statistics about checkpoints in directory.

        Returns:
            Dictionary with checkpoint stats
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint*.json"))

        total_size = sum(f.stat().st_size for f in checkpoint_files)

        return {
            "num_checkpoints": len(checkpoint_files),
            "total_size_mb": total_size / (1024 * 1024),
            "checkpoint_dir": str(self.checkpoint_dir),
            "oldest": min((f.stat().st_mtime for f in checkpoint_files), default=None),
            "newest": max((f.stat().st_mtime for f in checkpoint_files), default=None),
        }


def should_resume(args) -> bool:
    """
    Determine whether to resume from checkpoint.

    Args:
        args: Command-line arguments

    Returns:
        True if should resume
    """
    resume_flag = getattr(args, "resume", False)
    checkpoint_path = getattr(args, "checkpoint_path", None)

    if not resume_flag:
        return False

    if checkpoint_path and Path(checkpoint_path).exists():
        return True

    # Look for latest checkpoint in default location
    checkpoint_dir = Path("output")
    if checkpoint_dir.exists():
        latest = max(
            (f for f in checkpoint_dir.glob("checkpoint_*.json")),
            key=lambda p: p.stat().st_mtime,
            default=None
        )
        return latest is not None

    return False


def merge_checkpoint_with_new_data(
    checkpoint_data: List[Dict],
    new_data: List[Dict],
    key: str = "question_id"
) -> List[Dict]:
    """
    Merge checkpoint results with new data being processed.

    Args:
        checkpoint_data: Data from checkpoint
        new_data: New data to process
        key: Key to use for deduplication

    Returns:
        Merged data
    """
    # Create lookup of checkpoint items
    checkpoint_items = {item.get(key): item for item in checkpoint_data}

    # Add/update with new data
    for item in new_data:
        item_key = item.get(key)
        if item_key and item_key in checkpoint_items:
            # Merge: keep checkpoint but update with new data
            checkpoint_items[item_key].update(item)
        elif item_key:
            checkpoint_items[item_key] = item

    # Return as list
    return list(checkpoint_items.values())
