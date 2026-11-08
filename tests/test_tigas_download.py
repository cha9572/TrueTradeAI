"""Offline downloader regression tests, runnable under Python 3.14 without torch."""

import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from scripts import download_tigas as downloader


class DownloadUtilityTests(unittest.TestCase):
    def test_windows_annotation_normalization_and_traversal(self):
        self.assertEqual(downloader.normalized_annotation_path(r"images\ADM\0_real\x.JPEG"), "images/ADM/0_real/x.JPEG")
        for path in ["../x.jpg", "images/../x.jpg", "C:/images/x.jpg", "images//x.jpg"]:
            with self.assertRaises(ValueError):
                downloader.normalized_annotation_path(path)

    def test_exact_100_and_2000_quotas(self):
        ratios = {"train": .8, "val": .1, "test": .1}
        self.assertEqual(downloader.split_counts(100, ratios), {"train": 80, "val": 10, "test": 10})
        self.assertEqual(downloader.split_counts(2000, ratios), {"train": 1600, "val": 200, "test": 200})
        with self.assertRaises(ValueError):
            downloader.split_counts(100000, ratios)

    def test_expansion_keeps_rank_prefix(self):
        rows = [{"label": i % 2, "repo_path": f"train/images/group/{i}.jpg"} for i in range(30)]
        ranked = downloader.ranked_candidates(rows, 0, 42)
        self.assertEqual(ranked[:5], downloader.ranked_candidates(list(reversed(rows)), 0, 42)[:5])
        self.assertEqual(len(ranked), 15)

    def test_label_directory_disagreement_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            path.write_text("image_path,label\nimages/generator/0_real/x.jpg,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagreement"):
                downloader.read_annotations(path, "train")


class DownloadIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "data/tigas"
        self.payloads, self.annotations = {}, {}
        self.config = {"repo_id": "H1merka/TIGAS_dataset", "revision": "a" * 40, "seed": 42,
                       "output_dir": str(self.output), "per_class": 10,
                       "split_ratios": {"train": .8, "val": .1, "test": .1},
                       "max_image_mb": 1, "max_download_mb": 5, "retries": 0,
                       "max_failures": 2, "preview_count": 0}
        for split_index, split in enumerate(downloader.SPLITS):
            self.annotations[split] = []
            for label in [0, 1]:
                for index in range(12):
                    relative = f"images/group/{'0_real' if label == 0 else '1_fake'}/{index}.png"
                    repo = f"{split}/{relative}"
                    buffer = io.BytesIO()
                    Image.new("RGB", (8, 8), (split_index * 50, label * 100, index * 10)).save(buffer, format="PNG")
                    self.payloads[repo] = buffer.getvalue()
                    self.annotations[split].append({"repo_path": repo, "relative_path": relative, "label": label,
                                                    "upstream_split": split, "generator": "group"})
        payloads = self.payloads

        class FakeApi:
            def get_paths_info(self, paths, **kwargs):
                return [SimpleNamespace(path=p, size=len(payloads[p]), blob_id="unused",
                                        lfs=SimpleNamespace(sha256=hashlib.sha256(payloads[p]).hexdigest())) for p in paths]

        def fake_download(filename, local_dir, **kwargs):
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payloads[filename])
            return str(path)

        self.patches = [patch.object(downloader, "PROJECT_ROOT", self.root),
                        patch.object(downloader, "inspect_repository", return_value=self.annotations),
                        patch("huggingface_hub.HfApi", return_value=FakeApi()),
                        patch("huggingface_hub.hf_hub_download", side_effect=fake_download)]
        self.mocks = [p.start() for p in self.patches]

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temporary.cleanup()

    def run_download(self):
        # Keep test output short; production still reports all progress and outcomes.
        with patch("builtins.print"), patch("tqdm.tqdm", side_effect=lambda **kwargs: SimpleNamespace(update=lambda n: None, close=lambda: None)):
            return downloader.download_dataset(self.config)

    def test_download_reuse_labels_and_original_splits(self):
        first = self.run_download()
        self.assertEqual(first["events"]["downloaded"], 20)
        self.assertEqual(self.mocks[-1].call_count, 20)
        second = self.run_download()
        self.assertEqual(second["events"], {"reused": 20})
        self.assertEqual(second["reserved_image_transfer_bytes"], 0)
        self.assertEqual(self.mocks[-1].call_count, 20)
        with (self.output / "metadata.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            self.assertEqual(row["split"], row["upstream_split"])
            self.assertEqual(row["label"], "0" if "/0_real/" in row["repo_path"] else "1")
            self.assertEqual(row["fake_type"], "real" if row["label"] == "0" else "unknown")
            self.assertTrue((self.root / row["image_path"]).is_file())

    def test_corrupt_local_file_is_redownloaded(self):
        self.run_download()
        with (self.output / "metadata.csv").open(encoding="utf-8") as handle:
            first = next(csv.DictReader(handle))
        (self.root / first["image_path"]).write_bytes(b"corrupt")
        report = self.run_download()
        self.assertEqual(report["events"]["invalid_local"], 1)
        self.assertEqual(report["events"]["downloaded"], 1)
        self.assertEqual(report["events"]["reused"], 19)

    def test_failed_attempt_is_logged_and_replacement_keeps_quota(self):
        original = self.mocks[-1].side_effect
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("simulated network failure")
            return original(*args, **kwargs)

        self.mocks[-1].side_effect = fail_once
        report = self.run_download()
        self.assertTrue(report["complete"])
        self.assertEqual(report["successful_images"], 20)
        self.assertEqual(report["events"]["failed_attempt"], 1)
        self.assertEqual(report["events"]["failed"], 1)

    def test_failed_expansion_preserves_complete_metadata(self):
        self.run_download()
        previous = (self.output / "metadata.csv").read_bytes()
        self.config["per_class"] = 11
        self.config["max_failures"] = 1
        self.mocks[-1].side_effect = OSError("simulated offline")
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.run_download()
        self.assertEqual(previous, (self.output / "metadata.csv").read_bytes())
        self.assertTrue((self.output / "metadata.partial.csv").exists())
        report = json.loads((self.output / "download_summary.json").read_text())
        self.assertFalse(report["complete"])

    def test_exact_content_cannot_cross_splits(self):
        train_first = downloader.ranked_candidates(self.annotations["train"], 0, 42)[0]["repo_path"]
        val_first = downloader.ranked_candidates(self.annotations["val"], 0, 42)[0]["repo_path"]
        self.payloads[val_first] = self.payloads[train_first]
        report = self.run_download()
        self.assertTrue(report["complete"])
        self.assertEqual(report["events"]["skipped_duplicate"], 1)
        with (self.output / "metadata.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertNotIn(val_first, [r["repo_path"] for r in rows])
        ownership = {}
        for row in rows:
            previous = ownership.setdefault(row["source_id"], row["split"])
            self.assertEqual(previous, row["split"])

    def test_byte_budget_stops_before_image_request(self):
        self.config["max_download_mb"] = .000001
        with self.assertRaisesRegex(RuntimeError, "budget"):
            self.run_download()
        self.assertEqual(self.mocks[-1].call_count, 0)
        self.assertFalse((self.output / "metadata.csv").exists())


if __name__ == "__main__":
    unittest.main()
