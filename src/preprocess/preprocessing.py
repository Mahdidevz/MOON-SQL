"""Preprocessing pipeline for text-to-SQL datasets.

This module normalises SQL / NatSQL queries, extracts query skeletons,
retrieves relevant database content, and produces the JSON dataset consumed
by the SEA-SQL training and evaluation stages.
"""

import argparse
import concurrent.futures
import csv
import json
import os.path
import re
import shutil
from typing import Any, Dict, List

from add_content import get_database_matches_with_bm25
from sql_metadata import Parser
from tqdm import tqdm

MODULE = "[preprocess]"


def log(message: str) -> None:
    """Print a prefixed, flushed log message to stdout.

    Args:
        message: The text to log.
    """
    print(f"{MODULE} {message}", flush=True)


sql_keywords: List[str] = [
    "select",
    "from",
    "where",
    "group",
    "order",
    "limit",
    "intersect",
    "union",
    "except",
    "join",
    "on",
    "as",
    "not",
    "between",
    "in",
    "like",
    "is",
    "exists",
    "max",
    "min",
    "count",
    "sum",
    "avg",
    "and",
    "or",
    "desc",
    "asc",
]


def parse_option() -> argparse.Namespace:
    """Parse command-line arguments for the preprocessing pipeline.

    Returns:
        argparse.Namespace: Parsed options including ``mode``, ``table_path``,
            ``input_dataset_path``, ``natsql_dataset_path``,
            ``output_dataset_path``, ``db_path``, ``target_type``,
            ``process_num``, and ``temp_column_corpus_path``.
    """
    parser = argparse.ArgumentParser("")

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--table_path", type=str, default="./data/spider/tables.json")
    parser.add_argument(
        "--input_dataset_path",
        type=str,
        default="./data/spider/train_spider.json",
        help="""
            options:
                ./data/spider/train_spider.json
                ./data/spider/dev.json
            """,
    )
    parser.add_argument(
        "--natsql_dataset_path",
        type=str,
        default="./NatSQL/NatSQLv1_6/train_spider-natsql.json",
        help="""
            options:
                ./NatSQL/NatSQLv1_6/train_spider-natsql.json
                ./NatSQL/NatSQLv1_6/dev-natsql.json
            """,
    )
    parser.add_argument(
        "--output_dataset_path",
        type=str,
        default="./data/pre-processing/preprocessed_dataset.json",
        help="the filepath of preprocessed dataset.",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="./data/spider/database",
        help="the filepath of database.",
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="sql",
        help="sql or natsql.",
    )
    parser.add_argument("--process_num", type=int, default=2)
    parser.add_argument(
        "--temp_column_corpus_path",
        type=str,
        default="temp_column_corpus_path",
    )

    opt = parser.parse_args()

    return opt


def get_db_contents(
    question: str,
    table_name_original: str,
    column_names_original: List[str],
    db_id: str,
    db_path: str,
    temp_column_corpus_path: str,
) -> List[List[Any]]:
    """Retrieve the top database cell-value matches for each column.

    For every column in *column_names_original*, queries the BM-25 index for
    values that are relevant to *question* and returns at most two matches per
    column.

    Args:
        question: The natural-language question.
        table_name_original: The table whose columns are being searched.
        column_names_original: Column names within *table_name_original*.
        db_id: Database identifier (used to locate the ``.sqlite`` file).
        db_path: Root directory containing per-database subdirectories.
        temp_column_corpus_path: Temporary directory for the BM-25 corpus.

    Returns:
        List[List[Any]]: One inner list per column, each holding up to two
            matched cell values.
    """
    matched_contents: List[List[Any]] = []
    # extract matched contents for each column
    for column_name_original in column_names_original:
        # matches = get_database_matches(
        #     question,
        #     table_name_original,
        #     column_name_original,
        #     db_path + "/{}/{}.sqlite".format(db_id, db_id)
        # )
        # matches = sorted(matches)
        matches: List[Any] = []
        new_content = get_database_matches_with_bm25(
            question,
            table_name_original,
            column_name_original,
            db_path + "/{}/{}.sqlite".format(db_id, db_id),
            db_id,
            tmp_file=temp_column_corpus_path,
        )
        for e in new_content:
            if e not in matches:
                matches.append(e)
        matches = matches[:2]
        matched_contents.append(matches)

    return matched_contents


def get_db_schemas(
    all_db_infos: List[Dict[str, Any]],
    db_path: str,
) -> Dict[str, Any]:
    """Build a schema dictionary keyed by database ID.

    Processes every entry in *all_db_infos* and produces a mapping from
    ``db_id`` to a dict containing ``pk``, ``fk``, and ``schema_items``.

    Args:
        all_db_infos: Raw database schema records (e.g. loaded from
            ``tables.json``).
        db_path: Root directory containing per-database subdirectories.

    Returns:
        Dict[str, Any]: Mapping of ``db_id`` to a schema dict with keys
            ``pk``, ``fk``, and ``schema_items``.
    """
    db_schemas: Dict[str, Any] = {}

    for db in all_db_infos:
        table_names_original = db["table_names_original"]
        table_names = db["table_names"]
        column_names_original = db["column_names_original"]
        column_names = db["column_names"]
        column_types = db["column_types"]

        db_schemas[db["db_id"]] = {}

        primary_keys: List[Dict[str, str]] = []
        foreign_keys: List[Dict[str, str]] = []
        # record primary keys
        for pk_column_idx in db["primary_keys"]:
            if isinstance(pk_column_idx, int):
                pk_table_name_original = table_names_original[
                    column_names_original[pk_column_idx][0]
                ]
                pk_column_name_original = column_names_original[pk_column_idx][1]
                primary_keys.append(
                    {
                        "table_name_original": pk_table_name_original.lower(),
                        "column_name_original": pk_column_name_original.lower(),
                    }
                )
            else:
                for pk_column_idx_t in pk_column_idx:
                    pk_table_name_original = table_names_original[
                        column_names_original[pk_column_idx_t][0]
                    ]
                    pk_column_name_original = column_names_original[pk_column_idx_t][1]
                    primary_keys.append(
                        {
                            "table_name_original": pk_table_name_original.lower(),
                            "column_name_original": pk_column_name_original.lower(),
                        }
                    )

        db_schemas[db["db_id"]]["pk"] = primary_keys

        # record foreign keys
        for source_column_idx, target_column_idx in db["foreign_keys"]:
            fk_source_table_name_original = table_names_original[
                column_names_original[source_column_idx][0]
            ]
            fk_source_column_name_original = column_names_original[source_column_idx][1]

            fk_target_table_name_original = table_names_original[
                column_names_original[target_column_idx][0]
            ]
            fk_target_column_name_original = column_names_original[target_column_idx][1]

            foreign_keys.append(
                {
                    "source_table_name_original": fk_source_table_name_original.lower(),
                    "source_column_name_original": fk_source_column_name_original.lower(),
                    "target_table_name_original": fk_target_table_name_original.lower(),
                    "target_column_name_original": fk_target_column_name_original.lower(),
                }
            )
        db_schemas[db["db_id"]]["fk"] = foreign_keys

        db_schemas[db["db_id"]]["schema_items"] = []
        for idx, table_name_original in enumerate(table_names_original):
            column_names_original_list: List[str] = []
            column_names_list: List[str] = []
            column_types_list: List[str] = []

            for column_idx, (table_idx, column_name_original) in enumerate(
                column_names_original
            ):
                if idx == table_idx:
                    column_names_original_list.append(column_name_original.lower())
                    column_names_list.append(column_names[column_idx][1].lower())
                    column_types_list.append(column_types[column_idx])

            column_description_dict: Dict[str, str] = {}
            value_description_dict: Dict[str, str] = {}
            column_description_list: List[str] = []
            value_description_list: List[str] = []
            description_path = os.path.join(
                db_path,
                db["db_id"],
                "database_description",
                f"{table_name_original}.csv",
            )
            if os.path.exists(description_path):
                with open(description_path, mode="r", encoding="latin-1") as file:
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        if row[0] != "original_column_name":
                            column_description_dict[row[0].lower()] = row[2].replace(
                                "\n", "\t"
                            )
                            value_description_dict[row[0].lower()] = row[1].replace(
                                "\n", "\t"
                            )

            for column in column_names_original_list:
                if column_description_dict.get(column) is not None:
                    column_description_list.append(column_description_dict[column])
                else:
                    column_description_list.append("")
                if value_description_dict.get(column) is not None:
                    value_description_list.append(value_description_dict[column])
                else:
                    value_description_list.append("")

            db_schemas[db["db_id"]]["schema_items"].append(
                {
                    "table_name_original": table_name_original.lower(),
                    "table_name": table_names[idx].lower(),
                    "column_names": column_names_list,
                    "column_names_original": column_names_original_list,
                    "column_types": column_types_list,
                    "column_description": column_description_list,
                    "value_description": value_description_list,
                }
            )

    return db_schemas


def normalization(sql: str) -> str:
    """Normalise a SQL or NatSQL string into a canonical form.

    Applies a fixed sequence of transformations: whitespace normalisation,
    lower-casing (outside quoted strings), semicolon removal, double-to-single
    quote conversion, implicit ASC insertion, and table-alias removal.

    Args:
        sql: Raw SQL or NatSQL string.

    Returns:
        str: Normalised SQL string.
    """

    def white_space_fix(s: str) -> str:
        parsed_s = Parser(s)
        s = " ".join([token.value for token in parsed_s.tokens])
        return s

    # convert everything except text between single quotation marks to lower case
    def lower(s: str) -> str:
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()

            if char == "'":
                in_quotation = not in_quotation

        return out_s

    # remove ";"
    def remove_semicolon(s: str) -> str:
        if s.endswith(";"):
            s = s[:-1]
        return s

    # double quotation -> single quotation
    def double_to_single(s: str) -> str:
        return s.replace('"', "'")

    def add_asc(s: str) -> str:
        pattern = re.compile(
            r"order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)"
            r"(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*"
        )
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")

        return s

    def remove_table_alias(s: str) -> str:
        tables_aliases = Parser(s).tables_aliases
        new_tables_aliases = {}
        for i in range(1, 11):
            if "t{}".format(i) in tables_aliases.keys():
                new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]

        tables_aliases = new_tables_aliases
        for k, v in tables_aliases.items():
            s = s.replace("as " + k + " ", "")
            s = s.replace(k, v)

        return s

    return remove_table_alias(
        add_asc(lower(white_space_fix(double_to_single(remove_semicolon(sql)))))
    )


def extract_skeleton(sql: str, db_schema: Dict[str, Any]) -> str:
    """Replace all schema-specific tokens in *sql* with ``_`` placeholders.

    Masks table names, column names, string literals, integers, and floats so
    that only the structural SQL keywords remain.

    Args:
        sql: Normalised SQL or NatSQL string.
        db_schema: Schema dict for the relevant database, as returned by
            ``get_db_schemas``.

    Returns:
        str: SQL skeleton with concrete values replaced by ``_``.
    """
    table_names_original: List[str] = []
    table_dot_column_names_original: List[str] = []
    column_names_original: List[str] = []
    for table in db_schema["schema_items"]:
        table_name_original = table["table_name_original"]
        table_names_original.append(table_name_original)

        for column_name_original in ["*"] + table["column_names_original"]:
            table_dot_column_names_original.append(
                table_name_original + "." + column_name_original
            )
            column_names_original.append(column_name_original)

    parsed_sql = Parser(sql)
    new_sql_tokens: List[str] = []
    for token in parsed_sql.tokens:
        # mask table names
        if token.value in table_names_original:
            new_sql_tokens.append("_")
        # mask column names
        elif (
            token.value in column_names_original
            or token.value in table_dot_column_names_original
        ):
            new_sql_tokens.append("_")
        # mask string values
        elif token.value.startswith("'") and token.value.endswith("'"):
            new_sql_tokens.append("_")
        # mask positive int number
        elif token.value.isdigit():
            new_sql_tokens.append("_")
        # mask negative int number
        elif is_negative_int(token.value):
            new_sql_tokens.append("_")
        # mask float number
        elif is_float(token.value):
            new_sql_tokens.append("_")
        else:
            new_sql_tokens.append(token.value.strip())

    sql_skeleton = " ".join(new_sql_tokens)

    # remove JOIN ON keywords
    sql_skeleton = sql_skeleton.replace("on _ = _ and _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace("on _ = _ or _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace(" on _ = _", "")
    pattern3 = re.compile("_ (?:join _ ?)+")
    sql_skeleton = re.sub(pattern3, "_ ", sql_skeleton)

    # "_ , _ , ..., _" -> "_"
    while "_ , _" in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("_ , _", "_")

    # remove clauses in WHERE keywords
    ops = ["=", "!=", ">", ">=", "<", "<="]
    for op in ops:
        if "_ {} _".format(op) in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("_ {} _".format(op), "_")
    while "where _ and _" in sql_skeleton or "where _ or _" in sql_skeleton:
        if "where _ and _" in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ and _", "where _")
        if "where _ or _" in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ or _", "where _")

    # remove additional spaces in the skeleton
    while "  " in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("  ", " ")

    return sql_skeleton


def is_negative_int(string: str) -> bool:
    """Return ``True`` if *string* represents a negative integer.

    Args:
        string: Token value to test.

    Returns:
        bool: ``True`` when the string is a negative integer literal.
    """
    if string.startswith("-") and string[1:].isdigit():
        return True
    else:
        return False


def is_float(string: str) -> bool:
    """Return ``True`` if *string* represents a floating-point number.

    Args:
        string: Token value to test.

    Returns:
        bool: ``True`` when the string is a float literal (including negative
            floats).
    """
    if string.startswith("-"):
        string = string[1:]

    s = string.split(".")
    if len(s) > 2:
        return False
    else:
        for s_i in s:
            if not s_i.isdigit():
                return False
        return True


def preprocess_data(
    i: int,
    dataset: List[Dict[str, Any]],
    all_db_infos: List[Dict[str, Any]],
    mode: str,
    natsql_dataset_path: str,
    target_type: str,
    db_path: str,
    output_dataset_path: str,
    temp_column_corpus_path: str,
) -> None:
    """Preprocess one chunk of the dataset and write the result to disk.

    Skips processing if the output file for chunk *i* already exists.

    Args:
        i: Zero-based chunk index.
        dataset: Slice of the full dataset assigned to this chunk.
        all_db_infos: Complete list of database schema records.
        mode: One of ``"train"``, ``"eval"``, or ``"test"``.
        natsql_dataset_path: Path to the corresponding NatSQL dataset file.
        target_type: Either ``"sql"`` or ``"natsql"``.
        db_path: Root directory containing per-database subdirectories.
        output_dataset_path: Base path for the output JSON files.
        temp_column_corpus_path: Temporary directory for the BM-25 corpus.
    """
    if os.path.exists(get_output_name(output_dataset_path, i)):
        log(f"Chunk {i + 1}: output already exists, skipping")
        return None

    assert mode in ["train", "eval", "test"]

    if mode in ["train", "eval"] and target_type == "natsql":
        # only train_spider.json and dev.json have corresponding natsql dataset
        natsql_dataset = json.load(open(natsql_dataset_path))
    else:
        # empty natsql dataset
        natsql_dataset = [None for _ in range(len(dataset))]

    db_schemas = get_db_schemas(all_db_infos, db_path)

    preprocessed_dataset: List[Dict[str, Any]] = []
    total = len(dataset)
    log(f"Chunk {i + 1}: processing {total} samples")

    for local_idx, (natsql_data, data) in enumerate(
        tqdm(zip(natsql_dataset, dataset), total=total), start=1
    ):
        db_id = data["db_id"]
        if local_idx == 1 or local_idx == total or local_idx % 25 == 0:
            log(
                f"Chunk {i + 1}: processing sample {local_idx}/{total} for db_id={db_id}"
            )

        question = (
            data["question"]
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", "'")
            .replace("\u201d", "'")
            .strip()
        )

        if mode == "test":
            sql, norm_sql, sql_skeleton = "", "", ""
            sql_tokens: List[str] = []

            natsql, norm_natsql, natsql_skeleton = "", "", ""
            natsql_used_columns: List[str] = []
            natsql_tokens: List[str] = []
        else:
            if data.get("query") is not None:
                sql = data["query"].strip()
            elif data.get("SQL") is not None:
                sql = data["SQL"].strip()
            else:
                sql = ""
            norm_sql = normalization(sql).strip()
            sql_skeleton = extract_skeleton(norm_sql, db_schemas[db_id]).strip()
            sql_tokens = norm_sql.split()

            if natsql_data is not None:
                natsql = natsql_data["NatSQL"].strip()
                norm_natsql = normalization(natsql).strip()
                natsql_skeleton = extract_skeleton(
                    norm_natsql, db_schemas[db_id]
                ).strip()
                natsql_used_columns = [
                    token
                    for token in norm_natsql.split()
                    if "." in token and token != "@.@"
                ]
                natsql_tokens = []
                for token in norm_natsql.split():
                    # split table_name_original.column_name_original
                    if "." in token:
                        natsql_tokens.extend(token.split("."))
                    else:
                        natsql_tokens.append(token)
            else:
                natsql, norm_natsql, natsql_skeleton = "", "", ""
                natsql_used_columns, natsql_tokens = [], []

        preprocessed_data: Dict[str, Any] = {}
        preprocessed_data["question"] = question
        preprocessed_data["db_id"] = db_id

        preprocessed_data["sql"] = sql
        preprocessed_data["norm_sql"] = norm_sql
        preprocessed_data["sql_skeleton"] = sql_skeleton

        preprocessed_data["natsql"] = natsql
        preprocessed_data["norm_natsql"] = norm_natsql
        preprocessed_data["natsql_skeleton"] = natsql_skeleton

        preprocessed_data["db_schema"] = []
        preprocessed_data["pk"] = db_schemas[db_id]["pk"]
        preprocessed_data["fk"] = db_schemas[db_id]["fk"]
        preprocessed_data["table_labels"] = []
        preprocessed_data["column_labels"] = []

        # add database information (table name, column name, table_labels, column_labels, ...)
        for table in db_schemas[db_id]["schema_items"]:
            db_contents = get_db_contents(
                question,
                table["table_name_original"],
                table["column_names_original"],
                db_id,
                db_path,
                temp_column_corpus_path,
            )

            preprocessed_data["db_schema"].append(
                {
                    "table_name_original": table["table_name_original"],
                    "table_name": table["table_name"],
                    "column_names": table["column_names"],
                    "column_names_original": table["column_names_original"],
                    "column_types": table["column_types"],
                    "db_contents": db_contents,
                    "column_description": table["column_description"],
                    "value_description": table["value_description"],
                }
            )

            # extract table and column classification labels
            if target_type == "sql":
                if table["table_name_original"] in sql_tokens:  # for used tables
                    preprocessed_data["table_labels"].append(1)
                    column_labels: List[int] = []
                    for column_name_original in table["column_names_original"]:
                        if (
                            column_name_original in sql_tokens  # for used columns
                            or table["table_name_original"] + "." + column_name_original
                            in sql_tokens
                        ):
                            column_labels.append(1)
                        else:
                            column_labels.append(0)
                    preprocessed_data["column_labels"].append(column_labels)
                else:  # for unused tables and their columns
                    preprocessed_data["table_labels"].append(0)
                    preprocessed_data["column_labels"].append(
                        [0 for _ in range(len(table["column_names_original"]))]
                    )
            elif target_type == "natsql":
                if table["table_name_original"] in natsql_tokens:  # for used tables
                    preprocessed_data["table_labels"].append(1)
                    column_labels = []
                    for column_name_original in table["column_names_original"]:
                        if (
                            table["table_name_original"] + "." + column_name_original
                            in natsql_used_columns
                        ):  # for used columns
                            column_labels.append(1)
                        else:
                            column_labels.append(0)
                    preprocessed_data["column_labels"].append(column_labels)
                else:
                    preprocessed_data["table_labels"].append(0)
                    preprocessed_data["column_labels"].append(
                        [0 for _ in range(len(table["column_names_original"]))]
                    )
            else:
                raise ValueError("target_type should be ``sql'' or ``natsql''")

        preprocessed_dataset.append(preprocessed_data)

    with open(get_output_name(output_dataset_path, i), "w") as f:
        preprocessed_dataset_str = json.dumps(
            preprocessed_dataset, indent=2, ensure_ascii=False
        )
        f.write(preprocessed_dataset_str)
    log(
        f"Chunk {i + 1}: saved {len(preprocessed_dataset)} records to "
        f"{get_output_name(output_dataset_path, i)}"
    )


def get_output_name(path: str, idx: int) -> str:
    """Derive a chunk-specific output filename from a base path.

    Inserts *idx* immediately before the final file extension.  For example,
    ``"out/dataset.json"`` with ``idx=0`` becomes ``"out/dataset0.json"``.

    Args:
        path: Base output file path.
        idx: Zero-based chunk index to embed in the filename.

    Returns:
        str: Modified path with the chunk index inserted.
    """
    paths = path.split(".")
    paths[-2] = paths[-2] + str(idx)
    return ".".join(paths)


def preprocess_parallel(
    dataset: List[Dict[str, Any]],
    all_db_infos: List[Dict[str, Any]],
    mode: str,
    natsql_dataset_path: str,
    target_type: str,
    db_path: str,
    output_dataset_path: str,
    process_num: int,
    temp_column_corpus_path: str,
) -> List[Any]:
    """Preprocess *dataset* in parallel using a thread pool.

    Splits *dataset* into *process_num* roughly equal chunks and processes
    each chunk concurrently via :func:`preprocess_data`.

    Args:
        dataset: Full input dataset to preprocess.
        all_db_infos: Complete list of database schema records.
        mode: One of ``"train"``, ``"eval"``, or ``"test"``.
        natsql_dataset_path: Path to the corresponding NatSQL dataset file.
        target_type: Either ``"sql"`` or ``"natsql"``.
        db_path: Root directory containing per-database subdirectories.
        output_dataset_path: Base path for the per-chunk output JSON files.
        process_num: Number of parallel worker threads.
        temp_column_corpus_path: Temporary directory for the BM-25 corpus.

    Returns:
        List[Any]: List of future results (one per chunk).
    """
    contents: List[Any] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(process_num):
            start = i * len(dataset) // process_num
            end = min((i + 1) * len(dataset) // process_num, len(dataset))
            futures.append(
                executor.submit(
                    preprocess_data,
                    i,
                    dataset[start:end],
                    all_db_infos,
                    mode,
                    natsql_dataset_path,
                    target_type,
                    db_path,
                    output_dataset_path,
                    temp_column_corpus_path,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            content = future.result()
            contents.append(content)
    return contents


def main(opt: argparse.Namespace) -> None:
    """Run the full preprocessing pipeline and write the final dataset file.

    Skips all processing if the output file already exists.  Otherwise,
    preprocesses the input dataset in parallel, then merges the per-chunk
    output files into a single JSON file.

    Args:
        opt: Parsed command-line options from :func:`parse_option`.
    """
    log(
        f"Starting module with input={opt.input_dataset_path}, output={opt.output_dataset_path}"
    )
    if os.path.exists(opt.output_dataset_path):
        log(f"Output {opt.output_dataset_path} already exists, skipping")
        return None
    dataset = json.load(open(opt.input_dataset_path))
    all_db_infos = json.load(open(opt.table_path))
    process_num = opt.process_num
    mode = opt.mode
    db_path = opt.db_path
    target_type = opt.target_type
    output_dataset_path = opt.output_dataset_path
    natsql_dataset_path = opt.natsql_dataset_path
    temp_column_corpus_path = opt.temp_column_corpus_path

    preprocess_parallel(
        dataset,
        all_db_infos,
        mode,
        natsql_dataset_path,
        target_type,
        db_path,
        output_dataset_path,
        process_num,
        temp_column_corpus_path,
    )

    log(f"Merging {process_num} chunk files into {output_dataset_path}")
    data: List[Any] = []
    for i in range(process_num):
        with open(get_output_name(output_dataset_path, i), "r") as f:
            data.extend(json.load(f))
        os.remove(get_output_name(output_dataset_path, i))
    with open(output_dataset_path, "w") as f:
        json.dump(data, f)
    if os.path.exists(temp_column_corpus_path):
        shutil.rmtree(temp_column_corpus_path)
    log(f"Finished preprocessing; saved {len(data)} records to {output_dataset_path}")


if __name__ == "__main__":
    opt = parse_option()
    main(opt)
