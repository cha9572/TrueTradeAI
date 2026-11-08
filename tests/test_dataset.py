"""Test leakage protection and full-image loading with temporary fixtures."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset import ImageDataset, build_transform, read_metadata


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ["one.png", "two.png"]:
            Image.new("RGB", (100, 40), "white").save(self.root / name)
        self.rows = [["one.png", 0, "real", "s1", "train"], ["two.png", 1, "A", "s2", "val"]]

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, extra=None):
        df = pd.DataFrame(self.rows, columns=["image_path", "label", "fake_type", "source_id", "split"])
        if extra:
            for key, values in extra.items():
                df[key] = values
        csv = self.root / "metadata.csv"
        df.to_csv(csv, index=False)
        return read_metadata(csv, self.root, ["photographer_id", "item_id"])

    def test_source_leakage(self):
        self.rows[1][3] = "s1"
        with self.assertRaisesRegex(ValueError, "leakage"):
            self.read()

    def test_optional_group_leakage(self):
        with self.assertRaisesRegex(ValueError, "photographer_id"):
            self.read({"photographer_id": ["p1", "p1"]})

    def test_path_alias_duplicate(self):
        self.rows[1][0] = "./one.png"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.read()

    def test_invalid_label(self):
        self.rows[1][1] = 2
        with self.assertRaisesRegex(ValueError, "label"):
            self.read()

    def test_invalid_fake_type(self):
        self.rows[1][2] = "real"
        with self.assertRaisesRegex(ValueError, "fake_type"):
            self.read()

    def test_general_dataset_fake_type_unknown(self):
        self.rows[1][2] = "unknown"
        self.assertEqual(self.read().iloc[1].fake_type, "unknown")

    def test_corrupt_image(self):
        (self.root / "one.png").write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "Invalid image"):
            self.read()

    def test_full_image_loader(self):
        config = {"input_size": 64, "mean": [.485, .456, .406], "std": [.229, .224, .225]}
        transform = build_transform(config)
        item = ImageDataset(self.read(), "train", transform)[0]
        self.assertEqual(tuple(item["image"].shape), (3, 64, 64))
        self.assertEqual(item["source_id"], "s1")
        # Both outermost colored pixels survive fitting a wide image into a square.
        image = Image.new("RGB", (100, 40), "white")
        image.paste("red", (0, 0, 20, 40))
        image.paste("blue", (80, 0, 100, 40))
        tensor = transform(image)
        self.assertGreater(tensor[0, 32, 2].item(), tensor[2, 32, 2].item())
        self.assertGreater(tensor[2, 32, 61].item(), tensor[0, 32, 61].item())

    def test_windows_spawn_loader(self):
        config = {"input_size": 64, "mean": [.485, .456, .406], "std": [.229, .224, .225]}
        dataset = ImageDataset(self.read(), "val", build_transform(config))
        loader = DataLoader(dataset, batch_size=1, num_workers=1, multiprocessing_context="spawn")
        batches = list(loader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(tuple(batches[0]["image"].shape), (1, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
