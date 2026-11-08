"""Show balanced TIGAS samples with original labels and verified local paths."""

import argparse
import csv
import os
import random
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))


def visualize_samples(metadata, output, count=10, seed=42):
    """Verify against original annotations and render five Real/Fake pairs by default."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    from scripts.download_tigas import read_annotations
    if type(count) is not int or count < 2 or count % 2:
        raise ValueError("Preview count must be a positive even number >=2")
    with Path(metadata).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(seed)
    by_label = {}
    for label in [0, 1]:
        group = [r for r in rows if int(r["label"]) == label]
        if len(group) < count // 2:
            raise ValueError("Not enough images for balanced preview")
        rng.shuffle(group)
        by_label[label] = group[:count // 2]
    selected = [by_label[label][i] for i in range(count // 2) for label in [0, 1]]
    annotations = {}
    fig, axes = plt.subplots(count // 2, 2, figsize=(18, 5 * (count // 2)), squeeze=False)
    manifest = []
    try:
        for number, (ax, row) in enumerate(zip(axes.flat, selected), 1):
            split, label = row["upstream_split"], int(row["label"])
            if split != row["split"]:
                raise ValueError("Local split differs from original split")
            path = (PROJECT_ROOT / row["image_path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            # Recover Hub local_dir from the known split/images/... repository path.
            hub = path
            for _ in Path(row["repo_path"]).parts:
                hub = hub.parent
            key = (str(hub), split)
            if key not in annotations:
                annotations[key] = {r["repo_path"]: r["label"] for r in read_annotations(hub / split / "annotations01.csv", split)}
            if annotations[key].get(row["repo_path"]) != label:
                raise ValueError(f"Original annotation label mismatch: {row['repo_path']}")
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((1000, 700))
                ax.imshow(image)
            ax.set_title(f"#{number} {'Real' if label == 0 else 'Fake'} (label={label}) | original split={split}", fontsize=12)
            caption = "\n".join(textwrap.wrap(f"Local: {row['image_path']}", width=85))
            ax.set_xlabel(caption, fontsize=8, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            manifest.append({"sample_number": number, "label": label, "split": split,
                             "image_path": row["image_path"], "repo_path": row["repo_path"],
                             "path_exists": True, "original_label_matches": True})
        fig.suptitle("TIGAS subset: 5 Real + 5 Fake | labels/paths checked against original annotation CSVs" if count == 10 else "TIGAS verified balanced samples", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .98), h_pad=4)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=120)
        with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
    finally:
        plt.close(fig)
    print(f"Verified {count} image paths and original labels. Preview: {output}", flush=True)
    return manifest


def main():
    """Render samples independently from downloading; works without PyTorch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/tigas/metadata.csv")
    parser.add_argument("--output", default="outputs/tigas/samples.png")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        visualize_samples(args.metadata, args.output, args.count, args.seed)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
