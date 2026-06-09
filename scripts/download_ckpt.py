"""Download released GHM checkpoints from Hugging Face.

Run from the repository root with `python scripts/download_ckpt.py`. The script
stages `logs/{CLIP,CDM,VLM}` from the model repository into the local
`checkpoints/{CLIP,CDM,VLM}` directory consumed by figure evaluation scripts.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FAMILIES = ("CLIP", "CDM", "VLM")


def normalize_path(value: str | Path) -> Path:
    """Handle WSL-style paths passed to Windows Python."""
    value = str(value)
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 6:
        drive = value[5]
        tail = value[7:].replace("/", "\\")
        return Path(f"{drive.upper()}:\\{tail}")
    return Path(value)


def env_path(name: str, default: Path) -> Path:
    """Read a path env var with WSL-to-Windows normalization."""
    value = os.environ.get(name)
    if not value:
        return default
    return normalize_path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("HF_CHECKPOINT_REPO", "faro1219/multimodal-ghm"))
    parser.add_argument(
        "--cache-dir",
        type=normalize_path,
        default=env_path("HF_CHECKPOINT_CACHE", REPO_ROOT / "download" / "multimodal-ghm"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=normalize_path,
        default=env_path("CHECKPOINT_DIR", REPO_ROOT / "checkpoints"),
    )
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN", "0") == "1")
    parser.add_argument("--check-only", action="store_true", default=os.environ.get("CHECK_ONLY", "0") == "1")
    return parser.parse_args()


def main() -> int:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as exc:
        raise SystemExit(
            "ERROR: huggingface_hub is required. Run `uv sync` or install "
            "`huggingface-hub`.\n"
            f"Import error: {exc}"
        )

    args = parse_args()
    repo_id = args.repo
    cache_dir = args.cache_dir
    checkpoint_dir = args.checkpoint_dir
    dry_run = args.dry_run
    check_only = args.check_only

    print("==> Downloading released checkpoints")
    print(f"    Hugging Face repo: {repo_id}/logs")
    print(f"    Local cache:       {cache_dir}")
    print(f"    Output directory:  {checkpoint_dir}")

    remote_files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model")
    checkpoint_files = [
        path
        for path in remote_files
        if path.startswith("logs/") and path.endswith("/checkpoint.pth")
    ]
    families = sorted({path.split("/")[1] for path in checkpoint_files})
    missing = [family for family in REQUIRED_FAMILIES if family not in families]
    if not checkpoint_files or missing:
        raise SystemExit(
            "ERROR: checkpoint files were not found in the expected Hugging Face layout.\n"
            f"Expected: logs/{{{','.join(REQUIRED_FAMILIES)}}}/.../checkpoint.pth\n"
            f"Found families: {families or 'none'}"
        )

    print(
        f"Remote preflight: found {len(checkpoint_files)} checkpoint files "
        f"under logs/{{{', '.join(families)}}}."
    )

    if dry_run or check_only:
        mode = "DRY_RUN=1" if dry_run else "CHECK_ONLY=1"
        print(f"{mode}: no checkpoint files downloaded or staged.")
        return 0

    local_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["logs/**"],
            local_dir=cache_dir,
        )
    )

    logs_dir = local_dir / "logs"
    if not logs_dir.exists():
        raise SystemExit(f"ERROR: downloaded snapshot does not contain logs/: {logs_dir}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for family in REQUIRED_FAMILIES:
        src = logs_dir / family
        dst = checkpoint_dir / family
        if not src.exists():
            raise SystemExit(f"ERROR: missing logs/{family} in downloaded snapshot: {logs_dir}")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        staged.append(family)
        print(f"staged logs/{family} -> {dst}")

    print(f"Checkpoint download complete. Staged families: {', '.join(staged)}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
