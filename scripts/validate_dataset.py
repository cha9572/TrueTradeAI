"""Validate an entire metadata file before training."""

import argparse

import yaml
from pathlib import Path
from src.dataset import dataset_summary, read_metadata


def main():
    """Resolve config paths relative to the project containing configs/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    path = Path(args.config).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    project = path.parent.parent
    data = config["data"]
    try:
        df = read_metadata(project / data["metadata"], project / data["root"], data.get("leakage_columns", []), data.get("verify_images", True))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(dataset_summary(df))
    print(f"Validated {len(df)} images. Real=0; Fake (A/B/C/unknown)=1.")


if __name__ == "__main__":
    main()
