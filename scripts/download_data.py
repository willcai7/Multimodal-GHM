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
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_FILE = (
    "zsc_imagenet_val/"
    "RN50-quickgelu_cc12m_20251208_214315/"
    "similarities_targets.pt"
)
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def bool_env(name: str, default: bool = False) -> bool:
    """Read a portable boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_ENV_VALUES


def normalize_path(value: str | Path) -> Path:
    """Normalize user-provided local filesystem paths."""
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 6:
        drive = value[5]
        tail = value[7:].replace("/", "\\")
        return Path(f"{drive.upper()}:\\{tail}")
    return Path(value)


def normalize_repo_path(value: str | Path) -> str:
    """Normalize Hugging Face repository paths to slash-separated paths."""
    path = str(PurePosixPath(str(value).replace("\\", "/"))).lstrip("/")
    if path in {"", "."}:
        raise argparse.ArgumentTypeError("repository path must not be empty")
    return path


def env_path(name: str, default: Path) -> Path:
    """Read a path env var with WSL-to-Windows normalization."""
    value = os.environ.get(name)
    if not value:
        return default
    return normalize_path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("HF_IMAGENET_REPO", "BiasCLIP/BiasCLIP"))
    parser.add_argument(
        "--source-file",
        type=normalize_repo_path,
        default=normalize_repo_path(os.environ.get("HF_IMAGENET_FILE", DEFAULT_SOURCE_FILE)),
    )
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
    parser.add_argument("--force", action="store_true", default=bool_env("FORCE"))
    parser.add_argument("--dry-run", action="store_true", default=bool_env("DRY_RUN"))
    parser.add_argument("--check-only", action="store_true", default=bool_env("CHECK_ONLY"))
    return parser.parse_args()


def paths_refer_to_same_file(left: Path, right: Path) -> bool:
    """Return True when both paths point to the same existing file."""
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def stage_downloaded_file(src: Path, target_file: Path) -> str:
    """Stage one downloaded file by hard-linking when possible, copying otherwise."""
    if paths_refer_to_same_file(src, target_file):
        return "already in place"

    target_file.parent.mkdir(parents=True, exist_ok=True)
    if target_file.exists() and target_file.is_dir():
        raise SystemExit(f"ERROR: target file path is a directory: {target_file}")
    if target_file.exists() or target_file.is_symlink():
        target_file.unlink()

    try:
        os.link(src, target_file)
        return "hard-linked"
    except OSError:
        shutil.copy2(src, target_file)
        return "copied"


def main() -> int:
    args = parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as exc:
        raise SystemExit(
            "ERROR: huggingface_hub is required. Run `uv sync` or install "
            "`huggingface-hub`.\n"
            f"Import error: {exc}"
        )

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

    parent = str(PurePosixPath(source_file).parent)
    if parent == ".":
        parent = ""
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

    transfer = stage_downloaded_file(src, target_file)

    if not target_file.exists() or target_file.stat().st_size == 0:
        raise SystemExit(f"ERROR: target file was not created correctly: {target_file}")

    print(f"Downloaded {src}")
    print(f"{transfer} -> {target_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
