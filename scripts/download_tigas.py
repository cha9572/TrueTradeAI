"""Download a bounded, resumable TIGAS subset using individual Hub file requests.

Only three annotation CSVs and selected images are fetched. No snapshot_download,
git clone, dataset parquet conversion, or recursive image download is performed.
This module deliberately has no torch, pandas, numpy or datasets dependency.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SPLITS = ("train", "val", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["image_path", "label", "fake_type", "source_id", "split", "domain",
          "dataset", "upstream_split", "repo_path", "revision", "generator", "sha256", "width", "height"]
LOG_FIELDS = ["run_id", "timestamp", "repo_path", "label", "upstream_split", "status",
              "attempt", "size_bytes", "sha256", "error"]


def write_json(path, value):
    """Atomically replace a JSON report so interrupted writes cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows, fields=FIELDS):
    """Atomically write a UTF-8 CSV with portable forward-slash image paths."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def normalized_annotation_path(value):
    """Normalize upstream Windows paths, rejecting traversal and absolute paths."""
    text = value.replace("\\", "/")
    parts = text.split("/")
    if not text.startswith("images/") or any(p in {"", ".", ".."} or ":" in p for p in parts):
        raise ValueError(f"Unsafe or unexpected annotation path: {value!r}")
    path = PurePosixPath(text)
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported annotated image: {value}")
    return path.as_posix()


def read_annotations(path, split):
    """Preserve original numeric labels and validate directory label consistency."""
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image_path", "label"]:
            raise ValueError(f"Unexpected upstream annotation schema: {reader.fieldnames}")
        for row in reader:
            relative = normalized_annotation_path(row["image_path"])
            if row["label"] not in {"0", "1"}:
                raise ValueError(f"Invalid original TIGAS label: {row}")
            label = int(row["label"])
            parts = PurePosixPath(relative).parts
            if len(parts) < 4 or parts[2] != ("0_real" if label == 0 else "1_fake"):
                raise ValueError(f"Annotation label/directory disagreement: {row}")
            if relative in seen:
                raise ValueError(f"Duplicate upstream path: {split}/{relative}")
            seen.add(relative)
            rows.append({"repo_path": f"{split}/{relative}", "relative_path": relative,
                         "label": label, "upstream_split": split, "generator": parts[1]})
    return rows


def split_counts(per_class, ratios):
    """Allocate exact per-class totals without moving examples between splits."""
    if type(per_class) is not int or not 10 <= per_class <= 2000:
        raise ValueError("per_class must be an integer from 10 to 2000 (100 initially)")
    if set(ratios) != set(SPLITS) or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in ratios.values()):
        raise ValueError("Positive train/val/test split_ratios required")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split_ratios must sum to 1")
    exact = {s: per_class * ratios[s] for s in SPLITS}
    counts = {s: math.floor(exact[s]) for s in SPLITS}
    order = sorted(SPLITS, key=lambda s: (-(exact[s] - counts[s]), SPLITS.index(s)))
    for split in order[:per_class - sum(counts.values())]:
        counts[split] += 1
    if min(counts.values()) < 1:
        raise ValueError("Each upstream split must receive at least one image per class")
    return counts


def ranked_candidates(rows, label, seed):
    """Rank independently of requested count so expansion reuses earlier selections."""
    return sorted((r for r in rows if r["label"] == label),
                  key=lambda r: hashlib.sha256(f"{seed}:{r['repo_path']}".encode()).digest())


def verify_image(path, expected_size, expected_sha256=None, expected_git_blob=None):
    """Check actual bytes, Hub content hash and full Pillow decode before reuse."""
    from PIL import Image
    path = Path(path)
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError("Image missing or size differs from pinned Hub file")
    digest = hashlib.sha256()
    git_digest = hashlib.sha1(f"blob {expected_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            git_digest.update(block)
    sha = digest.hexdigest()
    if expected_sha256 and sha != expected_sha256:
        raise ValueError("SHA256 differs from pinned Hub file")
    if not expected_sha256 and expected_git_blob and git_digest.hexdigest() != expected_git_blob:
        raise ValueError("Git blob hash differs from pinned Hub file")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError("Empty image dimensions")
    return {"sha256": sha, "width": width, "height": height}


class EventLog:
    """Flush every outcome to an append-only CSV, including failed attempts."""

    def __init__(self, path, run_id):
        self.path, self.run_id = Path(path), run_id
        self.counts = Counter()
        self.handle = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=LOG_FIELDS)
        if self.path.stat().st_size == 0:
            self.writer.writeheader()

    def record(self, row, status, attempt=0, size=0, sha="", error=""):
        self.writer.writerow({"run_id": self.run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                              "repo_path": row["repo_path"], "label": row["label"],
                              "upstream_split": row["upstream_split"], "status": status,
                              "attempt": attempt, "size_bytes": size, "sha256": sha, "error": str(error)})
        self.handle.flush()
        self.counts[status] += 1

    def close(self):
        self.handle.close()


def existing_owners(log_path):
    """Recover content split ownership even after interruption or subset resizing."""
    owners = {}
    if Path(log_path).exists():
        with Path(log_path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["status"] in {"downloaded", "reused"} and row["sha256"]:
                    identity = (row["upstream_split"], int(row["label"]))
                    old = owners.setdefault(row["sha256"], identity)
                    if old != identity:
                        raise ValueError("Existing download log has conflicting content split/label ownership")
    return owners


def load_download_config(path, per_class=None):
    """Use JSON to keep the downloader independent from training YAML/PyTorch."""
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if per_class is not None:
        config["per_class"] = per_class
    if config["repo_id"] != "H1merka/TIGAS_dataset":
        raise ValueError("This downloader validates the H1merka/TIGAS_dataset layout only")
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Pin revision to an immutable 40-character commit SHA")
    split_counts(config["per_class"], config["split_ratios"])
    for key in ["max_image_mb", "max_download_mb"]:
        if not isinstance(config[key], (int, float)) or not 0 < config[key] <= 4096:
            raise ValueError(f"Invalid safety limit: {key}")
    if type(config["retries"]) is not int or not 0 <= config["retries"] <= 5:
        raise ValueError("retries must be between 0 and 5")
    if type(config["max_failures"]) is not int or config["max_failures"] < 1:
        raise ValueError("max_failures must be positive")
    for key in ["output_dir", "preview_path"]:
        config[key] = str((PROJECT_ROOT / config[key]).resolve())
    if not Path(config["output_dir"]).is_relative_to(PROJECT_ROOT):
        raise ValueError("output_dir must be inside the project for portable training CSV paths")
    return config


def inspect_repository(api, config, output):
    """Inspect shallow tree entries and fetch only the three small annotation CSVs."""
    from huggingface_hub import hf_hub_download
    common = {"repo_id": config["repo_id"], "repo_type": "dataset", "revision": config["revision"]}
    structure, annotations = {}, {}
    for location in [None, *SPLITS]:
        entries = list(api.list_repo_tree(path_in_repo=location, recursive=False, **common))
        structure[location or "/"] = [{"path": e.path, "size": getattr(e, "size", None), "type": type(e).__name__} for e in entries]
    for split in SPLITS:
        expected = {f"{split}/images", f"{split}/annotations01.csv"}
        if not expected.issubset({x["path"] for x in structure[split]}):
            raise ValueError(f"Pinned repository does not have expected {split} files")
        path = hf_hub_download(filename=f"{split}/annotations01.csv", local_dir=output / "hub", **common)
        annotations[split] = read_annotations(path, split)
    # Paths are compared without split prefixes: an annotation must never be shared.
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            if {r["relative_path"] for r in annotations[first]} & {r["relative_path"] for r in annotations[second]}:
                raise ValueError(f"Upstream annotation overlap: {first}/{second}")
    report = {"repo_id": config["repo_id"], "revision": config["revision"], "structure": structure,
              "annotation_counts": {s: {str(label): sum(r["label"] == label for r in rows) for label in [0, 1]} for s, rows in annotations.items()}}
    write_json(output / "repository_structure.json", report)
    return annotations


def download_dataset(config, inspect_only=False):
    """Download exact balanced quotas; retain upstream split, labels and provenance."""
    from huggingface_hub import HfApi, hf_hub_download
    from tqdm import tqdm
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "dataset_state.json"
    identity = {k: config[k] for k in ["repo_id", "revision", "seed"]}
    if state_path.exists() and json.loads(state_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Existing dataset has a different revision/seed. Use a separate output_dir.")
    write_json(state_path, identity)
    api = HfApi()
    annotations = inspect_repository(api, config, output)
    counts = split_counts(config["per_class"], config["split_ratios"])
    print("Pinned TIGAS revision:", config["revision"], flush=True)
    print("Per-class requested split counts:", counts, flush=True)
    if inspect_only:
        print("Inspected annotation CSVs only. No images downloaded.")
        return {"complete": True, "inspect_only": True}
    common = {"repo_id": config["repo_id"], "repo_type": "dataset", "revision": config["revision"]}
    log_path = output / "download_log.csv"
    owners = existing_owners(log_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log = EventLog(log_path, run_id)
    rows, remote_info, used_paths = [], {}, set()
    reserved_bytes, failures, error = 0, 0, None
    max_image = int(config["max_image_mb"] * 1024 ** 2)
    max_total = int(config["max_download_mb"] * 1024 ** 2)
    progress = tqdm(total=config["per_class"] * 2, desc="TIGAS selected images", unit="image")
    try:
        for split in SPLITS:
            for label in [0, 1]:
                candidates = ranked_candidates(annotations[split], label, config["seed"])
                successes = 0
                for index, candidate in enumerate(candidates):
                    if successes == counts[split]:
                        break
                    repo_path = candidate["repo_path"]
                    if repo_path not in remote_info:
                        # Inspect bounded batches of candidate file metadata, never full image trees.
                        batch = candidates[index:index + min(32, counts[split] - successes + 4)]
                        inspected = api.get_paths_info(paths=[r["repo_path"] for r in batch], **common)
                        remote_info.update({r.path: r for r in inspected})
                    info = remote_info.get(repo_path)
                    if info is None or not hasattr(info, "blob_id"):
                        log.record(candidate, "failed", error="Annotated image absent in pinned repository")
                        failures += 1
                    elif info.size > max_image:
                        log.record(candidate, "failed", size=info.size, error="Image exceeds max_image_mb; not downloaded")
                        failures += 1
                    else:
                        lfs_sha = info.lfs.sha256 if info.lfs else None
                        if lfs_sha in owners and owners[lfs_sha] != (split, label):
                            log.record(candidate, "skipped_duplicate", sha=lfs_sha, error="Exact content already owned by another split/label")
                            continue
                        local = output / "hub" / repo_path
                        valid, image_info = False, None
                        if local.exists():
                            try:
                                image_info = verify_image(local, info.size, lfs_sha, info.blob_id)
                                valid = True
                            except (OSError, ValueError, SyntaxError) as exc:
                                log.record(candidate, "invalid_local", size=info.size, error=exc)
                        status = "reused" if valid else "downloaded"
                        if not valid:
                            for attempt in range(1, config["retries"] + 2):
                                if reserved_bytes + info.size > max_total:
                                    raise RuntimeError("Image download byte budget exhausted; incomplete download, increase limit within 4096 MiB or reduce per_class")
                                reserved_bytes += info.size
                                try:
                                    filename = hf_hub_download(filename=repo_path, local_dir=output / "hub",
                                                               force_download=local.exists() or attempt > 1, **common)
                                    image_info = verify_image(filename, info.size, lfs_sha, info.blob_id)
                                    valid = True
                                    break
                                except Exception as exc:
                                    log.record(candidate, "failed_attempt", attempt, info.size, error=f"{type(exc).__name__}: {exc}")
                                    if attempt <= config["retries"]:
                                        time.sleep(min(attempt, 2))
                            if not valid:
                                log.record(candidate, "failed", error="All download/validation attempts failed")
                                failures += 1
                        if valid:
                            sha = image_info["sha256"]
                            if sha in owners and owners[sha] != (split, label):
                                log.record(candidate, "skipped_duplicate", sha=sha, error="Decoded file ownership conflicts with another split/label")
                                continue
                            owners[sha] = (split, label)
                            if repo_path in used_paths:
                                raise ValueError("Duplicate selected repo path")
                            used_paths.add(repo_path)
                            log.record(candidate, status, size=info.size, sha=sha)
                            rows.append({"image_path": local.relative_to(PROJECT_ROOT).as_posix(), "label": label,
                                         "fake_type": "real" if label == 0 else "unknown", "source_id": f"tigas:{sha}",
                                         "split": split, "domain": "general", "dataset": config["repo_id"],
                                         "upstream_split": split, "repo_path": repo_path, "revision": config["revision"],
                                         "generator": candidate["generator"], **image_info})
                            successes += 1
                            progress.update(1)
                    if failures >= config["max_failures"]:
                        raise RuntimeError(f"Stopped after {failures} failed files; see {log_path}")
                if successes != counts[split]:
                    raise RuntimeError(f"Not enough valid images in upstream {split}, label={label}")
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
        if "candidate" in locals():
            log.record(candidate, "run_aborted", error=error)
    finally:
        progress.close()
        log.close()
    complete = error is None and len(rows) == config["per_class"] * 2
    metadata = output / ("metadata.csv" if complete else "metadata.partial.csv")
    write_csv(metadata, rows)
    summary = {"run_id": run_id, "complete": complete, "repo_id": config["repo_id"], "revision": config["revision"],
               "requested_per_class": config["per_class"], "requested_per_split_per_class": counts,
               "successful_images": len(rows), "counts": {s: {str(label): sum(r["split"] == s and r["label"] == label for r in rows) for label in [0, 1]} for s in SPLITS},
               "events": dict(log.counts), "reserved_image_transfer_bytes": reserved_bytes,
               "metadata": str(metadata), "error": error}
    write_json(output / "download_summary.json", summary)
    write_json(output / "runs" / f"{run_id}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not complete:
        raise RuntimeError(f"Download incomplete; original complete metadata preserved. {error}")
    if config.get("preview_count", 10):
        from scripts.visualize_tigas_samples import visualize_samples
        visualize_samples(metadata, config["preview_path"], config["preview_count"], config["seed"])
    return summary


def main():
    """Default to 100 Real + 100 Fake; --per-class 2000 expands without redownload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tigas_download.json")
    parser.add_argument("--per-class", type=int)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    try:
        from filelock import FileLock
        config = load_download_config(args.config, args.per_class)
        Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        # A second Windows terminal must not interleave log/state/metadata writes.
        with FileLock(str(Path(config["output_dir"]) / ".download.lock"), timeout=0):
            download_dataset(config, args.inspect_only)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
