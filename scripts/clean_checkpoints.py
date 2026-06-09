#!/usr/bin/env python3
"""Remove unserializable or unnecessary loss objects from checkpoint files."""

import argparse
import sys
import os
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    import torch
except Exception as exc:
    print(f"ERROR: Failed to import torch: {exc}", file=sys.stderr)
    sys.exit(1)


DEFAULT_EXTENSIONS = {".pth", ".pt", ".ckpt"}


def iter_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    """Yield checkpoint-like files under ``root`` matching ``exts``."""
    lower_exts = {e.lower() for e in exts}
    for path in root.rglob("*"):
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in lower_exts:
                yield path


def ensure_backup_path(original: Path) -> Path:
    """Return a non-existing backup path next to ``original``."""
    backup = original.with_suffix(original.suffix + ".bak")
    if not backup.exists():
        return backup
    # Avoid overwriting existing backup; add numeric suffix
    i = 1
    while True:
        candidate = original.with_suffix(original.suffix + f".bak{i}")
        if not candidate.exists():
            return candidate
        i += 1


def remove_loss_key(checkpoint_path: Path, dry_run: bool, create_backup: bool) -> Tuple[bool, str]:
    """Remove the top-level ``loss`` entry from one PyTorch checkpoint."""
    try:
        # Safe to cpu
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"SKIP (load error): {checkpoint_path} -> {exc}"

    if not isinstance(ckpt, dict):
        return False, f"SKIP (not a dict): {checkpoint_path}"

    if "loss" not in ckpt:
        return False, f"OK (no 'loss'): {checkpoint_path}"

    # Remove and save back
    try:
        if dry_run:
            return True, f"WOULD REMOVE 'loss': {checkpoint_path}"

        del ckpt["loss"]

        if create_backup:
            backup_path = ensure_backup_path(checkpoint_path)
            try:
                # Copy bytes
                with open(checkpoint_path, "rb") as src, open(backup_path, "wb") as dst:
                    dst.write(src.read())
            except Exception as exc:
                return False, f"ERROR (backup failed): {checkpoint_path} -> {exc}"

        torch.save(ckpt, str(checkpoint_path))
        return True, f"REMOVED 'loss': {checkpoint_path}"
    except Exception as exc:
        return False, f"ERROR (save failed): {checkpoint_path} -> {exc}"


def main(argv: List[str]) -> int:
    """Parse CLI arguments and process all matching checkpoints."""
    parser = argparse.ArgumentParser(description="Remove top-level 'loss' key from PyTorch checkpoints.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "logs"),
        help="Root directory to scan (default: <repo>/logs)",
    )
    parser.add_argument(
        "--ext",
        dest="exts",
        action="append",
        default=None,
        help="File extension to include (e.g., .pth). May be used multiple times. Defaults to .pth,.pt,.ckpt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not modify files; only report what would change.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backups before overwriting files.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: Root directory does not exist or is not a directory: {root}", file=sys.stderr)
        return 2

    exts = set(DEFAULT_EXTENSIONS if args.exts is None else {e if e.startswith('.') else f'.{e}' for e in args.exts})

    targets = list(iter_files(root, exts))
    if not targets:
        print(f"No files found under {root} with extensions: {sorted(exts)}")
        return 0

    print(f"Scanning {len(targets)} files under {root} with extensions: {sorted(exts)}")
    num_changed = 0
    num_errors = 0

    for fp in targets:
        changed, msg = remove_loss_key(fp, dry_run=args.dry_run, create_backup=not args.no_backup)
        print(msg)
        if changed:
            num_changed += 1
        elif msg.startswith("ERROR"):
            num_errors += 1

    print(f"\nSummary: changed={num_changed}, errors={num_errors}, total={len(targets)}")
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


