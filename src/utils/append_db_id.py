"""Append database IDs to generated SQL predictions for evaluation.

This module combines SQL predictions with their corresponding database IDs,
preparing them for downstream evaluation against the BIRD benchmark dataset.
"""

import argparse
import json
from typing import Any, Dict, List, Optional

MODULE = "[append_db_id]"


def log(message: str) -> None:
    """Print log message with module name prefix.

    Args:
        message: Log message to print.
    """
    print(f"{MODULE} {message}", flush=True)


def parse_option() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with parsed arguments:
            - dev_path: Path to dev.json file
            - input_path: Path to SQL predictions file
            - output_path: Path to save output JSON
    """
    parser = argparse.ArgumentParser("Append database IDs to SQL predictions")

    parser.add_argument("--dev_path", type=str, default="dev.json",
                        help="Path to dev.json file with database IDs")
    parser.add_argument("--input_path", type=str,
                        default="../intermediate_datasets_bird/third_round.sql",
                        help="Path to file containing SQL predictions")
    parser.add_argument("--output_path", type=str,
                        default="../intermediate_datasets_bird/predict_dev.json",
                        help="Path to save output JSON with DB IDs")

    opt = parser.parse_args()

    return opt


def main(options: argparse.Namespace) -> None:
    """Main processing function.

    Loads SQL predictions and corresponding database IDs, combines them,
    and saves to output file.

    Args:
        options: Parsed command-line arguments.
    """
    log(f"Starting module with dev_path={options.dev_path}, input_path={options.input_path}, "
        f"output_path={options.output_path}")
    dev_path = options.dev_path
    output_path = options.output_path
    sql_path = options.input_path

    dev = json.load(open(dev_path))
    log(f"Loaded {len(dev)} dev records")
    sqls: List[str] = []
    with open(sql_path) as f:
        lines = f.readlines()
        for line in lines:
            sqls.append(line.strip())
    log(f"Loaded {len(sqls)} SQL statements")

    for i in range(len(sqls)):
        sqls[i] = (
            sqls[i].replace("|| ', ' ||", ", ")
            .replace("|| ' ' ||", ", ")
            + "\t----- bird -----\t"
            + dev[i]["db_id"]
        )
        if i == 0 or i == len(sqls) - 1 or (i + 1) % 25 == 0:
            log(f"Processing sample {i + 1}/{len(sqls)}")

    result: Dict[int, str] = {}
    for i, sql in enumerate(sqls):
        result[i] = sql

    if output_path:
        json.dump(result, open(output_path, "w"), indent=4)
        log(f"Finished module; saved {len(result)} records to {output_path}")


if __name__ == "__main__":
    opt = parse_option()
    main(opt)