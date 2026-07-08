#!/usr/bin/env python3
"""
Dynamic dataset slicer for the MOON-SQL pipeline.

Reads data/spider/dev.json, slices the first N samples, and writes:
  - data/spider/dev_subset.json          (pipeline input)
  - data/spider_data/dev_subset_gold.sql (gold labels for evaluation)

When --full is used (or no -n is supplied), ensures a full gold SQL file
exists at data/spider_data/dev_gold.sql and prints the full-dataset paths.

Outputs KEY=VALUE lines at the end so the calling shell script can parse
the resolved paths without hard-coding them.
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice Spider dev.json for token-efficient pipeline testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python slice_data.py -n 50      # use first 50 samples\n"
            "  python slice_data.py --full     # use all samples\n"
            "  python slice_data.py            # defaults to --full\n"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-n",
        "--num_samples",
        type=int,
        metavar="N",
        help="Number of samples to extract from dev.json (must be > 0).",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Use the complete dev.json dataset with no slicing.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> list:
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(data: list, path: Path) -> None:
    """Write a list to a JSON file with readable formatting."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def write_gold_sql(data: list, path: Path) -> None:
    """
    Write a gold SQL file in Spider evaluation format.
    Each line: <query>TAB<db_id>
    This matches what evaluation.py expects when it reads
    '{ground_truth_path}{data_mode}_gold.sql'.
    """
    with open(path, "w", encoding="utf-8") as fh:
        for entry in data:
            query = entry.get("query", "").strip().replace("\n", " ")
            db_id = entry.get("db_id", "").strip()
            if not query or not db_id:
                print(
                    f"[WARNING] Skipping entry with missing query or db_id: {entry}",
                    file=sys.stderr,
                )
                continue
            fh.write(f"{query}\t{db_id}\n")


def emit(key: str, value: str) -> None:
    """Print a KEY=VALUE line that the calling shell script can parse."""
    print(f"{key}={value}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ── Resolve project paths ──────────────────────────────────────────────
    # __file__ is src/utils/slice_data.py, so the project root is two levels up.
    project_root = Path(__file__).parent.parent.parent.resolve()
    dev_json_path = project_root / "data" / "spider" / "dev.json"
    subset_json_path = project_root / "data" / "spider" / "dev_subset.json"
    gold_dir = project_root / "data" / "spider_data"
    full_gold_path = gold_dir / "dev_gold.sql"
    subset_gold_path = gold_dir / "dev_subset_gold.sql"

    # ── Validate source file ───────────────────────────────────────────────
    if not dev_json_path.exists():
        print(f"[ERROR] Source file not found: {dev_json_path}", file=sys.stderr)
        sys.exit(1)

    full_data = load_json(dev_json_path)
    total = len(full_data)
    print(f"[INFO] Loaded {total} samples from {dev_json_path}")

    # ── Full-dataset mode ──────────────────────────────────────────────────
    if args.full or args.num_samples is None:
        print("[INFO] Mode: FULL — no slicing applied")

        if not full_gold_path.exists():
            print(f"[INFO] Generating full gold SQL → {full_gold_path}")
            write_gold_sql(full_data, full_gold_path)
        else:
            print(f"[INFO] Full gold SQL already exists: {full_gold_path}")

        # Emit resolved paths for the shell wrapper
        emit("DEV_PATH", str(dev_json_path))
        emit("DATA_MODE", "dev")
        return

    # ── Subset mode ────────────────────────────────────────────────────────
    n = args.num_samples

    if n <= 0:
        print(f"[ERROR] -n must be a positive integer, got {n}", file=sys.stderr)
        sys.exit(1)

    if n > total:
        print(
            f"[WARNING] Requested {n} samples but dev.json only has {total}. "
            "Using all available samples.",
            file=sys.stderr,
        )
        n = total

    subset = full_data[:n]
    print(f"[INFO] Mode: SUBSET — slicing first {n} of {total} samples")

    # Write subset JSON
    write_json(subset, subset_json_path)
    print(f"[INFO] Subset JSON written    → {subset_json_path}")

    # Write subset gold SQL
    write_gold_sql(subset, subset_gold_path)
    print(f"[INFO] Subset gold SQL written → {subset_gold_path}")

    # Emit resolved paths for the shell wrapper
    emit("DEV_PATH", str(subset_json_path))
    emit("DATA_MODE", "dev_subset")


if __name__ == "__main__":
    main()
