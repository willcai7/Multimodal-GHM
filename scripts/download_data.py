"""Download the ImageNet zero-shot tensor used by Fig. 7.

Run from the repository root with `python scripts/download_data.py`. The file is
hosted in the BiasCLIP dataset repository. It is large, so the script performs a
remote preflight and hard-links from the local Hugging Face snapshot cache to
the target path when possible.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_FILE = (
    "zsc_imagenet_val/"
    "RN50-quickgelu_cc12m_20251208_214315/"
    "similarities_targets.pt"
)


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
    parser.add_argument("--repo", default=os.environ.get("HF_IMAGENET_REPO", "BiasCLIP/BiasCLIP"))
    parser.add_argument("--source-file", default=os.environ.get("HF_IMAGENET_FILE", DEFAULT_SOURCE_FILE))
    parser.add_argument(
        "--cache-dir",
        type=normalize_path,
        default=env_path("HF_DATA_CACHE", REPO_ROOT / "download" / "BiasCLIP"),
    )
    parser.add_argument(
        "--target-file",
        type=normalize_path,
        default=env_path(
            "IMAGENET_TARGET",
            REPO_ROOT / "figures" / "data" / "imagenet-data" / "similarities_targets.pt",
        ),
    )
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE", "0") == "1")
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
    source_file = args.source_file
    cache_dir = args.cache_dir
    target_file = args.target_file
    force = args.force
    dry_run = args.dry_run
    check_only = args.check_only

    print("==> Downloading ImageNet Fig. 7 data")
    print(f"    Hugging Face dataset: {repo_id}")
    print(f"    Source file:          {source_file}")
    print(f"    Target file:          {target_file}")
    print(f"    Local cache:          {cache_dir}")

    if target_file.exists() and not force and not dry_run and not check_only:
        print("Target already exists. Set FORCE=1 to download again.")
        return 0

    parent = str(Path(source_file).parent).replace("\\", "/")
    try:
        entries = list(
            HfApi().list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=parent,
                recursive=True,
                expand=True,
            )
        )
    except Exception as exc:
        raise SystemExit(
            "ERROR: failed to inspect the BiasCLIP dataset. The dataset may "
            "require accepting access terms and authenticating with "
            "`huggingface-cli login` or HF_TOKEN.\n"
            f"Inspection error: {exc}"
        )

    source_entry = next((entry for entry in entries if getattr(entry, "path", None) == source_file), None)
    if source_entry is None:
        available = [getattr(entry, "path", "") for entry in entries]
        raise SystemExit(
            "ERROR: ImageNet source file not found in the BiasCLIP dataset.\n"
            f"Expected: {source_file}\n"
            f"Available under parent path: {available or 'none'}"
        )

    size_bytes = getattr(source_entry, "size", None)
    if size_bytes:
        print(f"Remote preflight: found {source_file} ({size_bytes / (1024 ** 3):.2f} GiB).")
    else:
        print(f"Remote preflight: found {source_file}.")

    if dry_run or check_only:
        mode = "DRY_RUN=1" if dry_run else "CHECK_ONLY=1"
        print(f"{mode}: no data file downloaded.")
        return 0

    try:
        local_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=[source_file],
                local_dir=cache_dir,
            )
        )
    except Exception as exc:
        raise SystemExit(
            "ERROR: failed to download the ImageNet data. The BiasCLIP dataset may "
            "require accepting access terms and authenticating with `huggingface-cli login` "
            "or HF_TOKEN.\n"
            f"Download error: {exc}"
        )

    src = local_dir / source_file
    if not src.exists():
        raise SystemExit(f"ERROR: downloaded file not found: {src}")

    target_file.parent.mkdir(parents=True, exist_ok=True)
    if target_file.exists():
        target_file.unlink()

    try:
        os.link(src, target_file)
        transfer = "hard-linked"
    except OSError:
        shutil.copy2(src, target_file)
        transfer = "copied"

    if not target_file.exists() or target_file.stat().st_size == 0:
        raise SystemExit(f"ERROR: target file was not created correctly: {target_file}")

    print(f"Downloaded {src}")
    print(f"{transfer} -> {target_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
