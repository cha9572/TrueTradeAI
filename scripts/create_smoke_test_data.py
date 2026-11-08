"""Create visibly synthetic fixtures ONLY for testing code, never model performance."""

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw


def create_data(output, root):
    """Generate split-disjoint source families with Real and Fake A/B/C labels."""
    output, root = Path(output).resolve(), Path(root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "metadata.csv"
    if metadata.exists():
        raise FileExistsError(f"Refusing to overwrite {metadata}; choose a new --output")
    rng = random.Random(42)
    rows = []
    for split in ["train", "val", "test"]:
        for family in range(2):
            source = f"smoke_{split}_{family}"
            for kind in ["real", "A", "B", "C"]:
                image = Image.new("RGB", (160, 100), tuple(rng.randrange(60, 220) for _ in range(3)))
                draw = ImageDraw.Draw(image)
                draw.rectangle((15, 10, 70, 85), outline="black", width=3)
                draw.text((75, 20), f"SYNTHETIC\n{split}\n{kind}", fill="black")
                path = output / f"{source}_{kind}.png"
                image.save(path)
                rows.append([path.relative_to(root).as_posix(), int(kind != "real"), kind, source, split, "smoke", "true"])
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label", "fake_type", "source_id", "split", "domain", "is_smoke"])
        writer.writerows(rows)
    (output / "SYNTHETIC_ONLY.txt").write_text("Code smoke test only. Not evidence of real-world detection performance.\n", encoding="utf-8")
    print(f"Created {len(rows)} SYNTHETIC pipeline fixtures: {metadata}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/smoke")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    create_data(args.output, args.root)
