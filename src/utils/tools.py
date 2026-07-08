"""Database schema and query utilities for SQL generation and analysis.

This module provides utilities for generating database schema representations,
extracting table and column information from SQL queries, and executing SQL queries
with timeout protection.
"""

import copy
import os
import queue
import re
import sqlite3
import sys
import threading
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from sql_metadata import Parser


def generate_foreign_key(data: Dict[str, Any]) -> str:
    """Generate a formatted string representation of foreign key relationships.

    Args:
        data: Dictionary containing 'fk' key with list of foreign key definitions.
              Each FK dict should have 'source_table_name_original',
              'source_column_name_original', 'target_table_name_original',
              and 'target_column_name_original' keys.

    Returns:
        A formatted string with foreign key relationships as comments.
    """
    fk_str = ""
    for fk in data["fk"]:
        fk_str += (
            f"# {fk['source_table_name_original']}.{fk['source_column_name_original']} = "
            f"{fk['target_table_name_original']}.{fk['target_column_name_original']}\n"
        )
    return fk_str[:-1]


def generate_schema(data: Dict[str, Any]) -> str:
    """Generate a formatted schema string from database structure.

    Args:
        data: Dictionary containing 'db_schema' key with list of table definitions.
              Each table dict should include 'table_name_original',
              'column_names_original', and 'db_contents'.

    Returns:
        A formatted schema string with table and column information.
    """
    schema = ""
    for table in data["db_schema"]:
        schema += f"# {table['table_name_original']} ( "
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


def generate_schema_simple(data: Dict[str, Any]) -> str:
    """Generate a simplified schema string without detailed content examples.

    Args:
        data: Dictionary containing 'db_schema' key with list of table definitions.

    Returns:
        A simplified formatted schema string.
    """
    schema = ""
    for table in data["db_schema"]:
        schema += f"# {table['table_name_original']} ( "
        for i, column in enumerate(table["column_names_original"]):
            schema += f"{column} ("
            schema += ", "
        schema = schema[:-2] + " )\n"
    return schema[:-1]


def generate_foreign_key_by_tables(
    data: Dict[str, Any], uses_tables: List[str]
) -> str:
    """Generate foreign key relationships for specified tables only.

    Args:
        data: Dictionary containing 'fk' key with foreign key definitions.
        uses_tables: List of table names to filter foreign keys by.

    Returns:
        A formatted string of foreign keys for specified tables.
    """
    fk_str = ""
    for fk in data["fk"]:
        if (
            fk["source_table_name_original"] in uses_tables
            and fk["target_table_name_original"] in uses_tables
        ):
            fk_str += (
                f"# {fk['source_table_name_original']}.{fk['source_column_name_original']} = "
                f"{fk['target_table_name_original']}.{fk['target_column_name_original']}\n"
            )
    return fk_str[:-1]


def generate_schema_by_tables(
    data: Dict[str, Any], uses_tables: List[str]
) -> str:
    """Generate schema for specified tables only.

    Args:
        data: Dictionary containing 'db_schema' key with table definitions.
        uses_tables: List of table names to include in schema.

    Returns:
        A formatted schema string for specified tables.
    """
    schema = ""
    for table in data["db_schema"]:
        if table["table_name_original"] in uses_tables:
            schema += f"# {table['table_name_original']} ( "
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


def get_subsets(items_list: List[str]) -> List[Tuple[str, ...]]:
    """Generate all subsets of size 2 and greater from a list.

    Args:
        items_list: List of items to generate subsets from.

    Returns:
        List of tuples representing subsets of size 2 and greater.
    """
    subsets = []
    for r in range(2, len(items_list) + 1):
        subsets.extend(combinations(items_list, r))
    return subsets


def find_path(
    graph: Dict[str, List[str]], start: str, end: str, path: Optional[List[str]] = None
) -> Optional[List[str]]:
    """Find a path between two nodes in a graph using depth-first search.

    Args:
        graph: Dictionary representing the graph adjacency list.
        start: Starting node name.
        end: Ending node name.
        path: Current path during recursion (default: None).

    Returns:
        List representing the path from start to end, or None if no path exists.
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


def get_tables(sql: str, data: Dict[str, Any]) -> List[str]:
    """Extract and resolve all tables referenced in a SQL query.

    This function identifies tables mentioned in SQL, adds related tables through
    foreign key relationships, and ensures table connectivity.

    Args:
        sql: SQL query string.
        data: Dictionary containing 'db_schema' and 'fk' keys.

    Returns:
        List of unique table names used in the query.
    """
    sql_lower = sql.lower()
    all_tables = [table["table_name_original"] for table in data["db_schema"]]
    sql_tokens = [s for s in sql_lower.replace("  ", " ").split(" ") if s.strip() != ""]
    uses_tables: List[str] = []
    end_punctuations = [
        s[-1] for s in all_tables if s and not s[-1].isalnum()
    ]
    sql_tokens = [
        s if (s[-1].isalnum() or s[-1] in end_punctuations) else s[:-1]
        for s in sql_tokens
    ]

    for table in all_tables:
        if table in sql_tokens:
            uses_tables.append(table)
    if len(uses_tables) == 0:
        uses_tables = all_tables

    new_tables = copy.deepcopy(uses_tables)
    for fk in data["fk"]:
        if fk["source_table_name_original"] in uses_tables:
            new_tables.append(fk["target_table_name_original"])
        elif fk["target_table_name_original"] in uses_tables:
            new_tables.append(fk["source_table_name_original"])
    uses_tables = list(set(new_tables))

    table_fk: List[List[str]] = []
    for fk in data["fk"]:
        source = fk["source_table_name_original"]
        target = fk["target_table_name_original"]
        table_fk.append([source, target])

    graph: Dict[str, List[str]] = {}
    for edge in table_fk:
        node1, node2 = edge
        if node1 in graph:
            graph[node1].append(node2)
        else:
            graph[node1] = [node2]
        if node2 in graph:
            graph[node2].append(node1)
        else:
            graph[node2] = [node1]

    if len(uses_tables) < 10:
        table_subsets = get_subsets(uses_tables)
        for pair in table_subsets:
            first = pair[0]
            second = pair[1]
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
    """Resolve related tables from a list using foreign key relationships.

    Args:
        uses_tables: List of table names to resolve.
        data: Dictionary containing 'db_schema' and 'fk' keys.

    Returns:
        List of resolved table names including related tables.
    """
    new_tables = copy.deepcopy(uses_tables)
    for fk in data["fk"]:
        if fk["source_table_name_original"] in uses_tables:
            new_tables.append(fk["target_table_name_original"])
        elif fk["target_table_name_original"] in uses_tables:
            new_tables.append(fk["source_table_name_original"])
    uses_tables = list(set(new_tables))

    table_fk: List[List[str]] = []
    for fk in data["fk"]:
        source = fk["source_table_name_original"]
        target = fk["target_table_name_original"]
        table_fk.append([source, target])

    graph: Dict[str, List[str]] = {}
    for edge in table_fk:
        node1, node2 = edge
        if node1 in graph:
            graph[node1].append(node2)
        else:
            graph[node1] = [node2]
        if node2 in graph:
            graph[node2].append(node1)
        else:
            graph[node2] = [node1]

    if len(uses_tables) < 10:
        table_subsets = get_subsets(uses_tables)
        for pair in table_subsets:
            first = pair[0]
            second = pair[1]
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


def normalization(sql: str) -> str:
    """Normalize SQL query for consistent processing.

    Applies multiple transformations: whitespace fixing, lowercasing (except quoted text),
    semicolon removal, quote normalization, and table alias removal.

    Args:
        sql: SQL query string to normalize.

    Returns:
        Normalized SQL query string.
    """

    def white_space_fix(query_str: str) -> str:
        """Fix irregular whitespace in SQL query."""
        parsed = Parser(query_str)
        return " ".join([token.value for token in parsed.tokens])

    def lower(query_str: str) -> str:
        """Convert to lowercase, preserving text in single quotes."""
        in_quotation = False
        out_str = ""
        for char in query_str:
            if in_quotation:
                out_str += char
            else:
                out_str += char.lower()

            if char == "'":
                in_quotation = not in_quotation

        return out_str

    def remove_semicolon(query_str: str) -> str:
        """Remove trailing semicolon from query."""
        if query_str.endswith(";"):
            query_str = query_str[:-1]
        return query_str

    def double_to_single(query_str: str) -> str:
        """Convert double quotes to single quotes."""
        return query_str.replace('"', "'")

    def add_asc(query_str: str) -> str:
        """Add ASC keyword to ORDER BY clauses that lack ASC/DESC."""
        pattern = re.compile(
            r"order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*"
        )
        if "order by" in query_str and "asc" not in query_str and "desc" not in query_str:
            for p_str in pattern.findall(query_str):
                query_str = query_str.replace(p_str, p_str + " asc")

        return query_str

    def remove_table_alias(query_str: str) -> str:
        """Remove table aliases from query and replace with original table names."""
        tables_aliases = Parser(query_str).tables_aliases
        new_tables_aliases = {}
        for i in range(1, 11):
            alias_key = f"t{i}"
            if alias_key in tables_aliases.keys():
                new_tables_aliases[alias_key] = tables_aliases[alias_key]

        tables_aliases = new_tables_aliases
        for key, value in tables_aliases.items():
            query_str = query_str.replace("as " + key + " ", "")
            query_str = query_str.replace(key, value)

        return query_str

    processing_func = lambda x: remove_table_alias(
        add_asc(lower(white_space_fix(double_to_single(remove_semicolon(x)))))
    )

    return processing_func(sql)


def get_table_columns(sql: str, data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract columns used from each table in a SQL query.

    Args:
        sql: SQL query string.
        data: Dictionary containing 'db_schema' key.

    Returns:
        Dictionary mapping table names to lists of column names used.
    """
    norm_sql = normalization(sql).strip()
    sql_tokens = norm_sql.split()
    uses_dict: Dict[str, List[str]] = {}
    for table_def in data["db_schema"]:
        if table_def["table_name_original"] in sql_tokens:
            uses_dict[table_def["table_name_original"]] = []
            for column_name_original in table_def["column_names_original"]:
                if (
                    column_name_original in sql_tokens
                    or f"{table_def['table_name_original']}.{column_name_original}"
                    in sql_tokens
                ):
                    uses_dict[table_def["table_name_original"]].append(
                        column_name_original
                    )
    return uses_dict


def get_tables_columns_dict(sql: str, data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Get a dictionary of tables and their relevant columns from SQL query.

    Includes columns related through foreign keys and columns explicitly mentioned.

    Args:
        sql: SQL query string.
        data: Dictionary containing 'db_schema' and 'fk' keys.

    Returns:
        Dictionary mapping table names to lists of relevant column names.
    """
    sql_lower = sql.lower()
    all_tables = [table["table_name_original"] for table in data["db_schema"]]
    uses_tables: List[str] = []
    return_dict: Dict[str, List[str]] = {}

    for table in all_tables:
        if table in sql_lower:
            uses_tables.append(table)
    if len(uses_tables) == 0:
        uses_tables = all_tables

    for table_def in data["db_schema"]:
        table = table_def["table_name_original"]
        if table in uses_tables:
            return_dict[table] = []

    for fk in data["fk"]:
        if fk["source_table_name_original"] in uses_tables:
            source_table = fk["source_table_name_original"]
            source_column = fk["source_column_name_original"]
            if source_column not in return_dict[source_table]:
                return_dict[source_table].append(source_column)
            target_table = fk["target_table_name_original"]
            if target_table not in uses_tables:
                return_dict[target_table] = []
            target_column = fk["target_column_name_original"]
            if target_column not in return_dict[target_table]:
                return_dict[target_table].append(target_column)
        elif fk["target_table_name_original"] in uses_tables:
            target_table = fk["target_table_name_original"]
            target_column = fk["target_column_name_original"]
            if target_column not in return_dict[target_table]:
                return_dict[target_table].append(target_column)
            source_table = fk["source_table_name_original"]
            if source_table not in uses_tables:
                return_dict[source_table] = []
            source_column = fk["source_column_name_original"]
            if source_column not in return_dict[source_table]:
                return_dict[source_table].append(source_column)

    for table_def in data["db_schema"]:
        table = table_def["table_name_original"]
        columns = table_def["column_names_original"]
        if table in return_dict.keys():
            for column in columns:
                if column in sql_lower:
                    return_dict[table].append(column)
            if len(return_dict[table]) == 0:
                return_dict[table] = columns
            return_dict[table] = list(set(return_dict[table]))

    return return_dict


def get_tables_columns_dict_only(
    sql: str, data: Dict[str, Any]
) -> Dict[str, List[str]]:
    """Get columns for each table without foreign key expansion.

    Args:
        sql: SQL query string.
        data: Dictionary containing 'db_schema' key.

    Returns:
        Dictionary mapping table names to lists of column names.
    """
    sql_lower = sql.lower()
    all_tables = [table["table_name_original"] for table in data["db_schema"]]
    uses_tables: List[str] = []
    return_dict: Dict[str, List[str]] = {}

    for table in all_tables:
        if table in sql_lower:
            uses_tables.append(table)
    if len(uses_tables) == 0:
        uses_tables = all_tables

    for table_def in data["db_schema"]:
        table = table_def["table_name_original"]
        if table in uses_tables:
            return_dict[table] = []

    for table_def in data["db_schema"]:
        table = table_def["table_name_original"]
        columns = table_def["column_names_original"]
        if table in return_dict.keys():
            for column in columns:
                if column in sql_lower:
                    return_dict[table].append(column)
            if len(return_dict[table]) == 0:
                return_dict[table] = columns
            return_dict[table] = list(set(return_dict[table]))

    return return_dict


def generate_schema_by_dict_only(
    data: Dict[str, Any], use_dict: Dict[str, List[str]]
) -> str:
    """Generate schema string for specific columns in each table.

    Args:
        data: Dictionary containing 'db_schema' key.
        use_dict: Dictionary mapping table names to column names to include.

    Returns:
        A formatted schema string with only specified columns.
    """
    schema = ""
    for table in data["db_schema"]:
        if table["table_name_original"] in use_dict.keys():
            schema += f"# {table['table_name_original']} ( "
            for i, column in enumerate(table["column_names_original"]):
                if column not in use_dict[table["table_name_original"]]:
                    continue
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
            schema = schema[:-3] + "\n"
    return schema[:-1]


def run_sql(database_path: str, sql_query: str) -> Optional[Exception]:
    """Execute SQL query against SQLite database.

    Args:
        database_path: Path to SQLite database file.
        sql_query: SQL query string to execute.

    Returns:
        Exception object if execution failed, None on success.
    """
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        return None
    except Exception as e:
        return e


class QueryThread(threading.Thread):
    """Thread for executing database queries with timeout support."""

    def stop(self) -> None:
        """Stop the thread gracefully."""
        lock = self._tstate_lock
        if lock is not None:
            assert not lock.locked()
        self._is_stopped = True
        self._tstate_lock = None
        if not self.daemon:
            import _thread

            _allocate_lock = _thread.allocate_lock
            _shutdown_locks_lock = _allocate_lock()
            _shutdown_locks: Set[Any] = set()
            with _shutdown_locks_lock:
                _shutdown_locks.discard(lock)


def new_run_sql(database_path: str, sql_query: str) -> Tuple[Any, bool]:
    """Execute SQL query with timeout protection.

    Args:
        database_path: Path to SQLite database file.
        sql_query: SQL query string to execute.

    Returns:
        Tuple of (result, success_flag). On success, result is up to 10 rows.
        On failure or timeout, result is error message or exception.
    """
    result_queue: queue.Queue = queue.Queue()

    def execute_query() -> None:
        """Execute query and place result in queue."""
        try:
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            p_res = cursor.fetchall()
            result_queue.put((p_res[:10], True))
        except Exception as e:
            result_queue.put((e, False))

    query_thread = QueryThread(target=execute_query)
    query_thread.start()
    query_thread.join(timeout=30)

    if query_thread.is_alive():
        return ("Execution timeout", False)

    return result_queue.get()


def generate_schema_list(data: Dict[str, Any]) -> str:
    """Generate a numbered list of all table columns in schema.

    Args:
        data: Dictionary containing 'db_schema' key.

    Returns:
        A formatted string with numbered column list.
    """
    schema = ""
    num = 1
    for table in data["db_schema"]:
        for i, column in enumerate(table["column_names_original"]):
            schema += f"# [{num}]. {table['table_name_original']}.{column}\n"
            num += 1
    return schema[:-1]


def generate_schema_list_all(
    data: Dict[str, Any], knowledge: Optional[Dict[str, str]] = None
) -> List[str]:
    """Generate detailed schema information for all columns with optional knowledge.

    Args:
        data: Dictionary containing 'db_schema' key with table definitions.
        knowledge: Optional dictionary with external knowledge for columns.

    Returns:
        List of schema description strings for each column.
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
    """Generate schema string sorted by table and column from usage dictionary.

    Args:
        data: Dictionary containing 'db_schema' key.
        use_dict: Dictionary mapping table names to column lists.

    Returns:
        A formatted schema string with sample values.
    """
    schema = ""
    for table in use_dict.keys():
        schema += f"# {table} ( "
        for column in use_dict[table]:
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


def get_foreign_keys_list(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract and deduplicate all foreign key columns by table.

    Args:
        data: Dictionary containing 'fk' and 'db_schema' keys.

    Returns:
        Dictionary mapping table names to lists of foreign key columns.
    """
    foreign_dict: Dict[str, List[str]] = {}
    for fk_def in data["fk"]:
        source_table = fk_def["source_table_name_original"]
        source_column = fk_def["source_column_name_original"]
        if source_table not in foreign_dict:
            foreign_dict[source_table] = [source_column]
        else:
            foreign_dict[source_table].append(source_column)

        target_table = fk_def["target_table_name_original"]
        target_column = fk_def["target_column_name_original"]
        if target_table not in foreign_dict:
            foreign_dict[target_table] = [target_column]
        else:
            foreign_dict[target_table].append(target_column)

    for source_table_def in data["db_schema"]:
        source_table = source_table_def["table_name_original"]
        for column in source_table_def["column_names_original"]:
            for target_table_def in data["db_schema"]:
                target_table = target_table_def["table_name_original"]
                if source_table != target_table:
                    for target_column in target_table_def["column_names_original"]:
                        if column == target_column:
                            if source_table not in foreign_dict:
                                foreign_dict[source_table] = [column]
                            else:
                                foreign_dict[source_table].append(column)
                            if target_table not in foreign_dict:
                                foreign_dict[target_table] = [target_column]
                            else:
                                foreign_dict[target_table].append(target_column)

    return foreign_dict


def get_primary_keys_list(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract primary key columns by table.

    Args:
        data: Dictionary containing 'pk' key with primary key definitions.

    Returns:
        Dictionary mapping table names to lists of primary key columns.
    """
    primary_keys_dict: Dict[str, List[str]] = {}
    for pk_def in data["pk"]:
        table_name = pk_def["table_name_original"]
        column_name = pk_def["column_name_original"]
        primary_keys_dict[table_name] = [column_name]
    return primary_keys_dict
