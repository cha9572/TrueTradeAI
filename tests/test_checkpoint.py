"""Test safe checkpoint loading and held-out identity protection."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from src.utils import check_held_out, identity_hashes, load_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_missing_checkpoint_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "Train the model first"):
                load_checkpoint(Path(tmp) / "missing.pth")

    def test_untrained_checkpoint_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.pth"
            torch.save({"model_state": {}}, path)
            with self.assertRaisesRegex(ValueError, "Not a trained"):
                load_checkpoint(path)

    def test_smoke_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smoke.pth"
            torch.save({"model_state": {}, "config": {}, "threshold": .5, "epoch": 1,
                        "optimizer_updates": 1, "is_smoke": True, "protected_identities": {}}, path)
            with self.assertRaisesRegex(ValueError, "SYNTHETIC"):
                load_checkpoint(path)
            self.assertTrue(load_checkpoint(path, allow_smoke=True)["is_smoke"])

    def test_changed_split_cannot_hide_seen_source(self):
        train = pd.DataFrame({"source_id": ["scene1"], "_resolved_path": ["train.png"]})
        checkpoint = {"protected_identities": identity_hashes(train, [])}
        test = pd.DataFrame({"source_id": ["scene1"], "_resolved_path": ["new.png"]})
        with self.assertRaisesRegex(ValueError, "source_id"):
            check_held_out(test, checkpoint)

    def test_seen_optional_group_cannot_hide(self):
        train = pd.DataFrame({"source_id": ["s1"], "item_id": ["i1"], "_resolved_path": ["train.png"]})
        checkpoint = {"protected_identities": identity_hashes(train, ["item_id"])}
        test = pd.DataFrame({"source_id": ["s2"], "item_id": ["i1"], "_resolved_path": ["test.png"]})
        with self.assertRaisesRegex(ValueError, "item_id"):
            check_held_out(test, checkpoint)

    def test_disjoint_sources_accepted(self):
        train = pd.DataFrame({"source_id": ["s1"], "_resolved_path": ["train.png"]})
        test = pd.DataFrame({"source_id": ["s2"], "_resolved_path": ["test.png"]})
        check_held_out(test, {"protected_identities": identity_hashes(train, [])})

    def test_dropping_group_column_cannot_bypass_protection(self):
        train = pd.DataFrame({"source_id": ["s1"], "item_id": ["i1"], "_resolved_path": ["train.png"]})
        test = pd.DataFrame({"source_id": ["s2"], "_resolved_path": ["test.png"]})
        with self.assertRaisesRegex(ValueError, "missing identity columns"):
            check_held_out(test, {"protected_identities": identity_hashes(train, ["item_id"])})


if __name__ == "__main__":
    unittest.main()
