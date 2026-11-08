"""CSV validation and full-image preprocessing; no region detection or cropping."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

REQUIRED = ["image_path", "label", "fake_type", "source_id", "split"]
SPLITS = {"train", "val", "test"}


def read_metadata(csv_path, root, leakage_columns=(), verify_images=True):
    """Validate labels, image files, duplicates and group separation across splits."""
    csv_path, root = Path(csv_path), Path(root).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {csv_path}. Supply real data or run scripts.create_smoke_test_data.")
    text_columns = [c for c in REQUIRED if c != "label"] + list(leakage_columns) + ["domain"]
    df = pd.read_csv(csv_path, dtype={c: str for c in text_columns}, keep_default_na=False)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Metadata has no images. Add real data or generate the explicitly synthetic smoke dataset.")
    for col in REQUIRED:
        if df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Empty required value: {col}")
    labels = pd.to_numeric(df.label, errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError("label must be Real=0 or Fake=1")
    df["label"] = labels.astype(int)
    if not df.split.isin(SPLITS).all():
        raise ValueError("split must be train, val, or test")
    valid_types = ((df.label == 0) & (df.fake_type == "real")) | ((df.label == 1) & df.fake_type.isin(["A", "B", "C", "unknown"]))
    if not valid_types.all():
        raise ValueError("Use fake_type=real for label 0 and A/B/C/unknown for label 1")
    for col in dict.fromkeys(["source_id", *leakage_columns]):
        if col not in df:
            continue
        values = df[col].astype(str).str.strip()
        if col == "source_id" and values.eq("").any():
            raise ValueError("source_id cannot be blank")
        present = df.loc[values.ne("")].copy()
        present[col] = values[values.ne("")]
        if (present.groupby(col).split.nunique() > 1).any():
            raise ValueError(f"Data leakage: {col} occurs in multiple splits")
        df[col] = values
    paths = [(root / p).resolve() for p in df.image_path]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate image paths (including resolved path aliases) are not allowed")
    for path in paths:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"Unsupported image format: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        if verify_images:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"Invalid image: {path}: {exc}") from exc
    df["_resolved_path"] = [str(p) for p in paths]
    return df


def dataset_summary(df):
    """Return split/class/type counts, including absent types."""
    return pd.crosstab(df.split, df.fake_type).reindex(index=["train", "val", "test"], columns=["real", "A", "B", "C", "unknown"], fill_value=0).fillna(0).astype(int)


class FullImageTransform:
    """Letterbox the entire RGB image, then normalize; augment only training images."""

    def __init__(self, size, mean, std, train=False):
        self.size = int(size)
        if self.size < 32:
            raise ValueError("input_size must be >=32")
        self.fill = tuple(round(255 * x) for x in mean)
        self.augment = transforms.RandomHorizontalFlip(p=0.5 if train else 0.0)
        self.tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __call__(self, image):
        """Apply EXIF orientation and preserve all visible image content."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BICUBIC, color=self.fill)
        return self.tensor(self.augment(image))


def build_transform(model_config, train=False):
    """Build the same deterministic preprocessing used at evaluation and inference."""
    return FullImageTransform(model_config["input_size"], model_config["mean"], model_config["std"], train)


class ImageDataset(Dataset):
    """Return full image tensors, binary labels and evaluation metadata."""

    def __init__(self, df, split, transform):
        self.frame = df.loc[df.split == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No images in split: {split}")
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row._resolved_path) as image:
            tensor = self.transform(image)
        result = {"image": tensor, "label": torch.tensor(float(row.label)), "image_path": row.image_path,
                  "fake_type": row.fake_type, "source_id": row.source_id}
        if "domain" in row.index:
            result["domain"] = str(row["domain"])
        return result
