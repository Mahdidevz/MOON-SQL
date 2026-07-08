"""
Utility functions for schema rendering, SQL normalisation, table/column
extraction, and database query execution used during training data generation.
"""

from __future__ import annotations

import copy
import queue
import re
import sqlite3
import threading
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

from sql_metadata import Parser

# ---------------------------------------------------------------------------
# Schema and foreign-key rendering
# ---------------------------------------------------------------------------


def generate_foreign_key(data: Dict[str, Any]) -> str:
    """Build a comment-prefixed foreign-key string from database schema data.

    Args:
        data: Database schema dictionary containing a ``fk`` key with a list
              of foreign-key relationship mappings.

    Returns:
        A newline-delimited string where each line describes one relationship
        in the form ``# source_table.source_col = target_table.target_col``.
    """
    fk_str = ""
    for fk in data["fk"]:
        fk_str += (
            f"# {fk['source_table_name_original']}.{fk['source_column_name_original']}"
            f" = {fk['target_table_name_original']}.{fk['target_column_name_original']}\n"
        )
    return fk_str[:-1]


def generate_schema(data: Dict[str, Any]) -> str:
    """Build a comment-prefixed schema string with inline sample values.

    Each table is rendered as a single line listing its column names alongside
    a few representative database-content examples.

    Args:
        data: Database schema dictionary with a ``db_schema`` key.

    Returns:
        A newline-delimited schema string, one table per line.
    """
    schema = ""
    for table in data["db_schema"]:
        schema += "# " + table["table_name_original"] + " ( "
        for i, column in enumerate(table["column_names_original"]):
            schema += f"{column} ("
            left_parenthesis = False
            if table["db_contents"][i]:
                value_flag = False
                schema += " e.g., `"
                for value in table["db_contents"][i]:
                    if len(str(value)) < 100:
                        schema += str(value) + "`, `"
                        value_flag = True
                if value_flag:
                    schema = schema[:-3] + ", etc. )"
                elif left_parenthesis:
                    schema = schema[:-8] + " )"
                else:
                    schema = schema[:-10]
            schema += ", "
        schema = schema[:-2] + " )\n"
    return schema[:-1]


def generate_foreign_key_by_tables(data: Dict[str, Any], uses_tables: List[str]) -> str:
    """Build a foreign-key string restricted to a specific set of tables.

    Args:
        data: Database schema dictionary containing a ``fk`` key.
        uses_tables: Table names to include in the output.

    Returns:
        A newline-delimited foreign-key string filtered to ``uses_tables``.
    """
    fk_str = ""
    for fk in data["fk"]:
        if (
            fk["source_table_name_original"] in uses_tables
            and fk["target_table_name_original"] in uses_tables
        ):
            fk_str += (
                f"# {fk['source_table_name_original']}.{fk['source_column_name_original']}"
                f" = {fk['target_table_name_original']}.{fk['target_column_name_original']}\n"
            )
    return fk_str[:-1]


def generate_schema_by_tables(data: Dict[str, Any], uses_tables: List[str]) -> str:
    """Build a schema string restricted to a specific set of tables.

    Args:
        data: Database schema dictionary with a ``db_schema`` key.
        uses_tables: Table names to include in the output.

    Returns:
        A newline-delimited schema string containing only the specified tables.
    """
    schema = ""
    for table in data["db_schema"]:
        if table["table_name_original"] in uses_tables:
            schema += "# " + table["table_name_original"] + " ( "
            for i, column in enumerate(table["column_names_original"]):
                schema += column
                if table["db_contents"][i]:
                    value_flag = False
                    schema += " ( e.g., `"
                    for value in table["db_contents"][i]:
                        if len(str(value)) < 100:
                            schema += str(value) + "`, `"
                            value_flag = True
                    if value_flag:
                        schema = schema[:-3] + ", etc. )"
                    else:
                        schema = schema[:-10]
                schema += ", "
            schema = schema[:-2] + " )\n"
    return schema[:-1]


def generate_schema_by_dict_only(
    data: Dict[str, Any], use_dict: Dict[str, List[str]]
) -> str:
    """Build a schema string for tables present in a use-dictionary.

    Args:
        data: Database schema dictionary with a ``db_schema`` key.
        use_dict: Dictionary mapping table names to column-name lists, as
                  returned by :func:`get_tables_columns_dict_only`.

    Returns:
        A newline-delimited schema string covering only the tables in
        ``use_dict``.
    """
    schema = ""
    for table in data["db_schema"]:
        if table["table_name_original"] in use_dict:
            flag = False
            schema += "# " + table["table_name_original"] + " ( "
            for i, column in enumerate(table["column_names_original"]):
                schema += f"{column} ("
                left_parenthesis = False
                if table["db_contents"][i]:
                    value_flag = False
                    schema += " e.g., `"
                    for value in table["db_contents"][i]:
                        if len(str(value)) < 100:
                            schema += str(value) + "`, `"
                            value_flag = True
                    if value_flag:
                        schema = schema[:-3] + ", etc. )"
                    elif left_parenthesis:
                        schema = schema[:-8] + " )"
                    else:
                        schema = schema[:-10]
            if flag:
                schema = schema[:-2] + " )\n"
            else:
                schema = schema[:-3] + "\n"
    return schema[:-1]


def generate_schema_list(data: Dict[str, Any]) -> str:
    """Generate a numbered list of all ``table.column`` pairs in the schema.

    Args:
        data: Database schema dictionary with a ``db_schema`` key.

    Returns:
        A newline-delimited string with one ``# [N]. table.column`` entry
        per column across all tables.
    """
    schema = ""
    num = 1
    for table in data["db_schema"]:
        for column in table["column_names_original"]:
            schema += f"# [{num}]. {table['table_name_original']}.{column}\n"
            num += 1
    return schema[:-1]


def generate_schema_list_all(
    data: Dict[str, Any], knowledge: Optional[str] = None
) -> List[str]:
    """Generate a detailed per-column schema list with optional external knowledge.

    Each element describes a single column and may include its description,
    value description, sample values, and any matching external knowledge.

    Args:
        data: Database schema dictionary with a ``db_schema`` key.
        knowledge: Optional external knowledge string; when provided it is
                   appended to entries whose column name appears in it.

    Returns:
        A list of strings, one per column, each describing that column in
        natural-language format.
    """
    schema: List[str] = []
    for table in data["db_schema"]:
        for i, column in enumerate(table["column_names_original"]):
            temp_schema = f"table: {table['table_name_original']}, column: {column}"
            if table["column_description"][i] != "":
                temp_schema += f", column description: {table['column_description'][i]}"
            if table["value_description"][i] != "":
                temp_schema += f", value description: {table['value_description'][i]}"
            if table["db_contents"][i]:
                use_flag = False
                temp_schema += ", sample value: "
                for value in table["db_contents"][i]:
                    if len(str(value)) < 100:
                        temp_schema += f"{str(value)}, "
                        use_flag = True
                if use_flag:
                    temp_schema = temp_schema[:-2]
                else:
                    temp_schema = temp_schema[:-16]
            if knowledge is not None and column in knowledge:
                temp_schema += f", external knowledge: {knowledge}"
            schema.append(temp_schema)
    return schema


def generate_schema_by_dict_sort(
    data: Dict[str, Any], use_dict: Dict[str, List[str]]
) -> str:
    """Build a schema string ordered by the columns specified in a dictionary.

    Args:
        data: Database schema dictionary with a ``db_schema`` key.
        use_dict: Ordered mapping of table names to column-name lists.

    Returns:
        A newline-delimited schema string respecting the column ordering in
        ``use_dict``.
    """
    schema = ""
    for table, columns in use_dict.items():
        schema += "# " + table + " ( "
        for column in columns:
            schema += column
            for table_source in data["db_schema"]:
                if table_source["table_name_original"] == table:
                    for i, column_source in enumerate(
                        table_source["column_names_original"]
                    ):
                        if column_source == column and table_source["db_contents"][i]:
                            schema += " ( e.g., `"
                            for value in table_source["db_contents"][i]:
                                if len(str(value)) < 100:
                                    schema += str(value) + "`, `"
                            schema = schema[:-3] + ", etc. )"
            schema += ", "
        schema = schema[:-2] + " )\n"
    return schema[:-1]


# ---------------------------------------------------------------------------
# Graph helpers for FK-aware table expansion
# ---------------------------------------------------------------------------


def get_subsets(lst: List[Any]) -> List[Tuple[Any, ...]]:
    """Return all subsets of length >= 2 from a list.

    Args:
        lst: Input list of elements.

    Returns:
        A list of tuples, each representing a subset of ``lst`` containing at
        least two elements.
    """
    subsets: List[Tuple[Any, ...]] = []
    for r in range(2, len(lst) + 1):
        subsets.extend(combinations(lst, r))
    return subsets


def find_path(
    graph: Dict[str, List[str]],
    start: str,
    end: str,
    path: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """Find a path between two nodes in an undirected graph using DFS.

    Args:
        graph: Adjacency-list representation of the graph.
        start: Starting node name.
        end: Target node name.
        path: Current traversal path; used internally during recursion.

    Returns:
        A list of node names forming the path from ``start`` to ``end``,
        or ``None`` if no path exists.
    """
    if path is None:
        path = []
    path = path + [start]
    if start == end:
        return path
    if start not in graph:
        return None
    for node in graph[start]:
        if node not in path:
            new_path = find_path(graph, node, end, path)
            if new_path:
                return new_path
    return None


# ---------------------------------------------------------------------------
# Table extraction from SQL
# ---------------------------------------------------------------------------


def get_tables(sql: str, data: Dict[str, Any]) -> List[str]:
    """Extract the set of relevant tables referenced by a SQL query.

    Beyond tables explicitly named in the query, the result is expanded to
    include tables transitively linked via foreign-key relationships.

    Args:
        sql: Raw SQL query string.
        data: Database schema dictionary with ``db_schema`` and ``fk`` keys.

    Returns:
        A deduplicated list of table names relevant to the query.
    """
    sql = sql.lower()
    all_tables = [table["table_name_original"] for table in data["db_schema"]]
    sql_tokens = [s for s in sql.replace("  ", " ").split(" ") if s.strip() != ""]
    uses_tables: List[str] = []
    end_punctuations = [s[-1] for s in all_tables if s and not s[-1].isalnum()]
    sql_tokens = [
        s if (s[-1].isalnum() or s[-1] in end_punctuations) else s[:-1]
        for s in sql_tokens
    ]

    for table in all_tables:
        if table in sql_tokens:
            uses_tables.append(table)
    if not uses_tables:
        uses_tables = all_tables

    new_tables = copy.deepcopy(uses_tables)
    for fk in data["fk"]:
        if fk["source_table_name_original"] in uses_tables:
            new_tables.append(fk["target_table_name_original"])
        elif fk["target_table_name_original"] in uses_tables:
            new_tables.append(fk["source_table_name_original"])
    uses_tables = list(set(new_tables))

    table_fk: List[List[str]] = [
        [fk["source_table_name_original"], fk["target_table_name_original"]]
        for fk in data["fk"]
    ]

    graph: Dict[str, List[str]] = {}
    for node1, node2 in table_fk:
        graph.setdefault(node1, []).append(node2)
        graph.setdefault(node2, []).append(node1)

    if len(uses_tables) < 10:
        for pair in get_subsets(uses_tables):
            first, second = pair[0], pair[1]
            pair_flag = True
            for fk in table_fk:
                if first in fk and second in fk:
                    pair_flag = False
            if pair_flag:
                path = find_path(graph, first, second)
                if path:
                    for p in path:
                        uses_tables.append(p)
                else:
                    for fk in table_fk:
                        if first in fk and second not in fk:
                            uses_tables.append(second)
                        elif first not in fk and second in fk:
                            uses_tables.append(first)

    return list(set(uses_tables))


def get_tables_from_dict(uses_tables: List[str], data: Dict[str, Any]) -> List[str]:
    """Expand a table list by including all foreign-key-linked neighbours.

    Mirrors the expansion logic in :func:`get_tables` but accepts an explicit
    list instead of a raw SQL string.

    Args:
        uses_tables: Initial list of table names.
        data: Database schema dictionary with ``fk`` key.

    Returns:
        A deduplicated, expanded list of table names.
    """
    new_tables = copy.deepcopy(uses_tables)
    for fk in data["fk"]:
        if fk["source_table_name_original"] in uses_tables:
            new_tables.append(fk["target_table_name_original"])
        elif fk["target_table_name_original"] in uses_tables:
            new_tables.append(fk["source_table_name_original"])
    uses_tables = list(set(new_tables))

    table_fk: List[List[str]] = [
        [fk["source_table_name_original"], fk["target_table_name_original"]]
        for fk in data["fk"]
    ]

    graph: Dict[str, List[str]] = {}
    for node1, node2 in table_fk:
        graph.setdefault(node1, []).append(node2)
        graph.setdefault(node2, []).append(node1)

    if len(uses_tables) < 10:
        for pair in get_subsets(uses_tables):
            first, second = pair[0], pair[1]
            pair_flag = True
            for fk in table_fk:
                if first in fk and second in fk:
                    pair_flag = False
            if pair_flag:
                path = find_path(graph, first, second)
                if path:
                    for p in path:
                        uses_tables.append(p)
                else:
                    for fk in table_fk:
                        if first in fk and second not in fk:
                            uses_tables.append(second)
                        elif first not in fk and second in fk:
                            uses_tables.append(first)

    return list(set(uses_tables))


# ---------------------------------------------------------------------------
# SQL normalisation
# ---------------------------------------------------------------------------


def normalization(sql: str) -> str:
    """Normalise a SQL string into a canonical form.

    Applies the following transformations in sequence:

    1. Remove trailing semicolons.
    2. Replace double quotes with single quotes.
    3. Tokenise and rejoin to normalise whitespace.
    4. Lower-case everything outside single-quoted literals.
    5. Append ``ASC`` to bare ``ORDER BY`` clauses.
    6. Expand ``T1``–``T10`` table aliases to their original table names.

    Args:
        sql: Raw SQL query string.

    Returns:
        The normalised SQL string.
    """

    def white_space_fix(s: str) -> str:
        """Tokenise and rejoin ``s`` to normalise internal whitespace."""
        parsed_s = Parser(s)
        return " ".join(token.value for token in parsed_s.tokens)

    def lower(s: str) -> str:
        """Lower-case all characters outside single-quoted string literals."""
        in_quotation = False
        out_s = ""
        for char in s:
            out_s += char if in_quotation else char.lower()
            if char == "'":
                in_quotation = not in_quotation
        return out_s

    def remove_semicolon(s: str) -> str:
        """Strip a trailing semicolon from ``s``."""
        return s[:-1] if s.endswith(";") else s

    def double_to_single(s: str) -> str:
        """Replace all double quotes with single quotes."""
        return s.replace('"', "'")

    def add_asc(s: str) -> str:
        """Append ``asc`` to ``ORDER BY`` clauses that lack a sort direction."""
        pattern = re.compile(
            r"order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)"
            r"(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*"
        )
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")
        return s

    def remove_table_alias(s: str) -> str:
        """Replace ``T1``–``T10`` aliases with their original table names."""
        tables_aliases = Parser(s).tables_aliases
        filtered_aliases = {
            f"t{i}": tables_aliases[f"t{i}"]
            for i in range(1, 11)
            if f"t{i}" in tables_aliases
        }
        for alias, table in filtered_aliases.items():
            s = s.replace("as " + alias + " ", "")
            s = s.replace(alias, table)
        return s

    return remove_table_alias(
        add_asc(lower(white_space_fix(double_to_single(remove_semicolon(sql)))))
    )


# ---------------------------------------------------------------------------
# Column extraction from SQL
# ---------------------------------------------------------------------------


def get_table_columns(sql: str, data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract the columns referenced per table in a SQL query.

    Args:
        sql: Raw SQL query string.
        data: Database schema dictionary with a ``db_schema`` key.

    Returns:
        A dictionary mapping each referenced table name to the list of its
        referenced column names.
    """
    norm_sql = normalization(sql).strip()
    sql_tokens = norm_sql.split()
    uses_dict: Dict[str, List[str]] = {}
    for d in data["db_schema"]:
        if d["table_name_original"] in sql_tokens:
            uses_dict[d["table_name_original"]] = []
            for column_name_original in d["column_names_original"]:
                if (
                    column_name_original in sql_tokens
                    or d["table_name_original"] + "." + column_name_original
                    in sql_tokens
                ):
                    uses_dict[d["table_name_original"]].append(column_name_original)
    return uses_dict


def get_tables_columns_dict(sql: str, data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build a table-to-columns mapping for tables referenced in a SQL query.

    Foreign-key-linked tables and their key columns are automatically included
    even when not directly mentioned in the query.

    Args:
        sql: Raw SQL query string.
        data: Database schema dictionary with ``db_schema`` and ``fk`` keys.

    Returns:
        A dictionary mapping table names to lists of relevant column names.
    """
    sql = sql.lower()
    all_tables = [table["table_name_original"] for table in data["db_schema"]]
    uses_tables: List[str] = []
    return_dict: Dict[str, List[str]] = {}

    for table in all_tables:
        if table in sql:
            uses_tables.append(table)
    if not uses_tables:
        uses_tables = all_tables

    for d in data["db_schema"]:
        table = d["table_name_original"]
        if table in uses_tables:
            return_dict[table] = []

    for fk in data["fk"]:
        if fk["source_table_name_original"] in uses_tables:
            if (
                fk["source_column_name_original"]
                not in return_dict[fk["source_table_name_original"]]
            ):
                return_dict[fk["source_table_name_original"]].append(
                    fk["source_column_name_original"]
                )
            if fk["target_table_name_original"] not in uses_tables:
                return_dict[fk["target_table_name_original"]] = []
            if (
                fk["target_column_name_original"]
                not in return_dict[fk["target_table_name_original"]]
            ):
                return_dict[fk["target_table_name_original"]].append(
                    fk["target_column_name_original"]
                )
        elif fk["target_table_name_original"] in uses_tables:
            if (
                fk["target_column_name_original"]
                not in return_dict[fk["target_table_name_original"]]
            ):
                return_dict[fk["target_table_name_original"]].append(
                    fk["target_column_name_original"]
                )
            if fk["source_table_name_original"] not in uses_tables:
                return_dict[fk["source_table_name_original"]] = []
            if (
                fk["source_column_name_original"]
                not in return_dict[fk["source_table_name_original"]]
            ):
                return_dict[fk["source_table_name_original"]].append(
                    fk["source_column_name_original"]
                )

    for d in data["db_schema"]:
        table = d["table_name_original"]
        columns = d["column_names_original"]
        if table in return_dict:
            for column in columns:
                if column in sql:
                    return_dict[table].append(column)
            if not return_dict[table]:
                return_dict[table] = columns
            return_dict[table] = list(set(return_dict[table]))

    return return_dict


def get_tables_columns_dict_only(
    sql: str, data: Dict[str, Any]
) -> Dict[str, List[str]]:
    """Build a table-to-columns mapping without foreign-key expansion.

    Unlike :func:`get_tables_columns_dict`, this function does not add
    foreign-key-linked tables or columns.

    Args:
        sql: Raw SQL query string.
        data: Database schema dictionary with a ``db_schema`` key.

    Returns:
        A dictionary mapping table names to lists of relevant column names.
    """
    sql = sql.lower()
    all_tables = [table["table_name_original"] for table in data["db_schema"]]
    uses_tables: List[str] = []
    return_dict: Dict[str, List[str]] = {}

    for table in all_tables:
        if table in sql:
            uses_tables.append(table)
    if not uses_tables:
        uses_tables = all_tables

    for d in data["db_schema"]:
        table = d["table_name_original"]
        if table in uses_tables:
            return_dict[table] = []

    for d in data["db_schema"]:
        table = d["table_name_original"]
        columns = d["column_names_original"]
        if table in return_dict:
            for column in columns:
                if column in sql:
                    return_dict[table].append(column)
            if not return_dict[table]:
                return_dict[table] = columns
            return_dict[table] = list(set(return_dict[table]))

    return return_dict


def generate_table_column_list(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build a mapping of table names to all their column names.

    Args:
        data: Database schema dictionary with a ``db_schema`` key.

    Returns:
        A dictionary mapping each table name to its ordered list of column
        names.
    """
    table_dict: Dict[str, List[str]] = {}
    for table in data["db_schema"]:
        table_name = table["table_name_original"]
        for column in table["column_names_original"]:
            table_dict.setdefault(table_name, []).append(column)
    return table_dict


# ---------------------------------------------------------------------------
# Database query execution
# ---------------------------------------------------------------------------


def run_sql(db: str, sql: str) -> Optional[Exception]:
    """Execute a SQL query against a SQLite database and return any error.

    Args:
        db: Path to the SQLite database file.
        sql: SQL query string to execute.

    Returns:
        ``None`` if the query executes successfully, or the raised
        :class:`Exception` if it fails.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        cursor.fetchall()
        return None
    except Exception as exc:
        return exc


class QueryThread(threading.Thread):
    """A :class:`threading.Thread` subclass that supports forced termination.

    Used by :func:`new_run_sql` to enforce a query execution timeout.
    """

    def stop(self) -> None:
        """Forcefully mark this thread as stopped and release its state lock.

        Manipulates CPython-internal thread state to terminate a thread that
        has exceeded its execution time budget.
        """
        lock = self._tstate_lock
        if lock is not None:
            assert not lock.locked()
        self._is_stopped = True
        self._tstate_lock = None
        if not self.daemon:
            import _thread

            _allocate_lock = _thread.allocate_lock
            _shutdown_locks_lock = _allocate_lock()
            _shutdown_locks: set = set()
            with _shutdown_locks_lock:
                _shutdown_locks.discard(lock)


def new_run_sql(
    db: str, sql: str
) -> Tuple[Union[List[Tuple[Any, ...]], Exception, str], bool]:
    """Execute a SQL query with a 30-second timeout.

    Spawns a :class:`QueryThread` and waits up to 30 seconds for a result.
    If the thread does not finish in time, a timeout message is returned.

    Args:
        db: Path to the SQLite database file.
        sql: SQL query string to execute.

    Returns:
        A two-tuple ``(result, success)`` where ``result`` is either the first
        10 rows of the query result (on success), the raised
        :class:`Exception` (on SQL error), or the string
        ``'Execution timeout'`` (on timeout); and ``success`` is ``True`` only
        when the query ran without error.
    """
    result_queue: queue.Queue = queue.Queue()

    def execute_query() -> None:
        """Run the query inside the worker thread and enqueue the outcome."""
        try:
            conn = sqlite3.connect(db)
            cursor = conn.cursor()
            cursor.execute(sql)
            p_res = cursor.fetchall()
            result_queue.put((p_res[:10], True))
        except Exception as exc:
            result_queue.put((exc, False))

    query_thread = QueryThread(target=execute_query)
    query_thread.start()
    query_thread.join(timeout=30)

    if query_thread.is_alive():
        return "Execution timeout", False

    return result_queue.get()
