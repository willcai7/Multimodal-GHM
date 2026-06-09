from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    script_path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


download_ckpt = load_script("download_ckpt")
download_data = load_script("download_data")


class DownloadScriptTests(unittest.TestCase):
    def test_bool_env_accepts_common_true_values(self):
        with mock.patch.dict(os.environ, {"DRY_RUN": "YeS", "CHECK_ONLY": "0"}, clear=False):
            self.assertTrue(download_ckpt.bool_env("DRY_RUN"))
            self.assertFalse(download_ckpt.bool_env("CHECK_ONLY"))
            self.assertTrue(download_data.bool_env("MISSING_FLAG", default=True))

    def test_normalize_repo_path_uses_posix_separators(self):
        self.assertEqual(download_data.normalize_repo_path(r"foo\bar\file.pt"), "foo/bar/file.pt")
        self.assertEqual(download_data.normalize_repo_path("/foo//bar/file.pt"), "foo/bar/file.pt")

    def test_stage_downloaded_file_copies_when_hardlink_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src = tmp_path / "source.pt"
            target = tmp_path / "nested" / "target.pt"
            src.write_text("payload", encoding="utf-8")

            def fail_link(_src, _dst):
                raise OSError("hard links unavailable")

            with mock.patch.object(download_data.os, "link", fail_link):
                transfer = download_data.stage_downloaded_file(src, target)

            self.assertEqual(transfer, "copied")
            self.assertEqual(target.read_text(encoding="utf-8"), "payload")

    def test_stage_downloaded_file_skips_same_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.pt"
            src.write_text("payload", encoding="utf-8")

            transfer = download_data.stage_downloaded_file(src, src)

            self.assertEqual(transfer, "already in place")
            self.assertEqual(src.read_text(encoding="utf-8"), "payload")


if __name__ == "__main__":
    unittest.main()
