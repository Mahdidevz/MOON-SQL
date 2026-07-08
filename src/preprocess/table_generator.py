"""Generate a ``tables.json`` schema file from a directory of SQLite databases.

This module walks a directory tree, extracts schema information from every
``.sqlite`` file it finds, and serialises the result as a JSON array that is
compatible with the Spider benchmark ``tables.json`` format.
"""

import argparse
import json
import os
import sqlite3
from typing import Any, Dict, List


def parse_option() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with ``database_path`` and
            ``table_path`` attributes.
    """
    parser = argparse.ArgumentParser("")

    parser.add_argument("--database_path", type=str, default="dev.json")
    parser.add_argument("--table_path", type=str, default="dev.json")

    opt = parser.parse_args()

    return opt


def traverse_directory_using_os(root_folder: str) -> List[str]:
    """Recursively collect all ``.sqlite`` file paths under *root_folder*.

    If *root_folder* is itself a file it is returned directly (provided it
    has the ``.sqlite`` extension).

    Args:
        root_folder: Path to a directory or a single ``.sqlite`` file.

    Returns:
        List[str]: Absolute or relative paths of every ``.sqlite`` file found.
    """
    file_list: List[str] = []
    if not os.path.isdir(root_folder):
        file_list.append(root_folder)
    else:
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                file_list.append(full_path)
    new_file_list: List[str] = []
    for f in file_list:
        if ".sqlite" in f:
            new_file_list.append(f)
    return new_file_list


def extract_schema(db_file: str) -> Dict[str, Any]:
    """Extract the schema of a SQLite database into a Spider-compatible dict.

    Reads table names, column names/types, primary keys, and foreign keys
    from the SQLite PRAGMA interfaces and returns them in the format expected
    by the Spider benchmark.

    Args:
        db_file: Filesystem path to the ``.sqlite`` database file.

    Returns:
        Dict[str, Any]: A dictionary with the following keys:
            ``column_names``, ``column_names_original``, ``column_types``,
            ``db_id``, ``foreign_keys``, ``primary_keys``,
            ``table_names``, ``table_names_original``.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Result dictionary following the Spider tables.json schema
    result: Dict[str, Any] = {
        "column_names": [],
        "column_names_original": [],
        "column_types": [],
        "db_id": db_file.split("/")[-1].split(".")[0],  # Use the filename stem as db_id
        "foreign_keys": [],
        "primary_keys": [],
        "table_names": [],
        "table_names_original": [],
    }

    # Retrieve all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    result["column_names"].append([-1, "*"])
    result["column_names_original"].append([-1, "*"])
    result["column_types"].append("text")
    for idx, table in enumerate(tables):
        table_name = table[0]
        result["table_names"].append(table_name)
        result["table_names_original"].append(table_name)

        # Retrieve column information
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        table_column_types: List[str] = []

        for col in columns:
            cid, name, type_, notnull, dflt_value, pk = col
            table_column_types.append(type_)

            # Record column name and its table index
            result["column_names"].append([idx, name])
            result["column_names_original"].append([idx, name])
            if pk:
                result["primary_keys"].append(len(result["column_names_original"]) - 1)

        result["column_types"].extend(table_column_types)

        # Retrieve foreign keys
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        foreign_keys = cursor.fetchall()
        for fk in foreign_keys:
            if len(fk) >= 4:  # Ensure the row has enough elements
                from_col = fk[3]
                table_to = fk[2]
                to_col = fk[4]
                result["foreign_keys"].append([table_name, from_col, table_to, to_col])

    for i in range(len(result["foreign_keys"])):
        from_table_idx = result["table_names"].index(result["foreign_keys"][i][0])
        to_table_idx = result["table_names"].index(result["foreign_keys"][i][2])

        from_column_idx = 0
        for j in range(len(result["column_names_original"])):
            if (
                result["column_names_original"][j][0] == from_table_idx
                and result["column_names_original"][j][1]
                == result["foreign_keys"][i][1]
            ):
                from_column_idx = j
            if (
                result["column_names_original"][j][0] == to_table_idx
                and result["column_names_original"][j][1]
                == result["foreign_keys"][i][3]
            ):
                to_table_idx = j
        result["foreign_keys"][i] = [from_column_idx, to_table_idx]

    conn.close()

    return result


def main(opt: argparse.Namespace) -> None:
    """Extract schemas from all databases and write the output ``tables.json``.

    Args:
        opt: Parsed command-line options with ``database_path`` and
            ``table_path`` attributes.
    """
    path = opt.database_path
    all_paths = traverse_directory_using_os(path)
    results: List[Dict[str, Any]] = []
    for p in all_paths:
        schema_info = extract_schema(p)
        results.append(schema_info)

    with open(opt.table_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    opt = parse_option()
    main(opt)
