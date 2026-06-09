"""Repository-relative paths shared by figure evaluation scripts."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"
GHM_DATA_DIR = REPO_ROOT / "figures" / "data" / "ghm-data"


def checkpoint_dir(model_family):
    """Return the checkpoint directory for a model family such as CLIP/CDM/VLM."""
    return CHECKPOINT_ROOT / model_family


def latest_checkpoint(path_run):
    """Return a run checkpoint, handling both flat and timestamped run folders."""
    direct_checkpoint = path_run / "checkpoint.pth"
    if direct_checkpoint.exists():
        return direct_checkpoint

    checkpoints = sorted(path_run.glob("*/checkpoint.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"Checkpoint file not found under: {path_run}")
    return checkpoints[-1]


def ghm_output_path(filename):
    """Return a path under ``figures/data/ghm-data`` and create the directory."""
    GHM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return GHM_DATA_DIR / filename
