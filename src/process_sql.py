"""SQL parsing and processing utilities for semantic SQL understanding.

This module provides tools for tokenizing, parsing, and processing SQL queries
into structured abstract syntax tree (AST) representations compatible with semantic
SQL generation models. It handles complex SQL constructs including joins, subqueries,
aggregations, and set operations (UNION, INTERSECT, EXCEPT).

Data Structure Documentation:
    - val: number (float), string (str), or sql (dict)
    - col_unit: (agg_id, col_id, is_distinct)
    - val_unit: (unit_op, col_unit1, col_unit2)
    - table_unit: (table_type, col_unit/sql)
    - cond_unit: (not_op, op_id, val_unit, val1, val2)
    - condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
    - sql: {
        'select': (is_distinct, [(agg_id, val_unit), ...]),
        'from': {'table_units': [table_unit1, ...], 'conds': condition},
        'where': condition,
        'groupBy': [col_unit1, ...],
        'orderBy': ('asc'/'desc', [val_unit1, ...]),
        'having': condition,
        'limit': int or None,
        'intersect': sql or None,
        'except': sql or None,
        'union': sql or None
      }

Assumptions:
    1. Input SQL is syntactically correct
    2. Only table names can have aliases (not columns)
    3. At most one set operation (INTERSECT/UNION/EXCEPT) per query level
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from nltk import word_tokenize

CLAUSE_KEYWORDS = (
    "select",
    "from",
    "where",
    "group",
    "order",
    "limit",
    "intersect",
    "union",
    "except",
)
JOIN_KEYWORDS = ("join", "on", "as")

WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
TABLE_TYPE = {
    "sql": "sql",
    "table_unit": "table_unit",
}

COND_OPS = ("and", "or")
SQL_OPS = ("intersect", "union", "except")
ORDER_OPS = ("desc", "asc")


class Schema:
    """Maps tables and columns to unique identifiers for SQL parsing.

    This class manages the mapping between original database table/column names
    and unique internal identifiers used throughout SQL parsing.
    """

    def __init__(self, schema: Dict[str, List[str]]) -> None:
        """Initialize Schema with database structure.

        Args:
            schema: Dictionary mapping table names to lists of column names.
        """
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self) -> Dict[str, List[str]]:
        """Get the database schema dictionary.

        Returns:
            Dictionary mapping table names to column lists.
        """
        return self._schema

    @property
    def idMap(self) -> Dict[str, str]:
        """Get the identifier mapping for tables and columns.

        Returns:
            Dictionary mapping table.column and table names to unique identifiers.
        """
        return self._idMap

    def _map(self, schema: Dict[str, List[str]]) -> Dict[str, str]:
        """Create mapping from table.column names to unique identifiers.

        Args:
            schema: Database schema dictionary.

        Returns:
            Dictionary with mapped identifiers.
        """
        id_map = {"*": "__all__"}
        for key, vals in schema.items():
            for val in vals:
                id_map[key.lower() + "." + val.lower()] = (
                    "__" + key.lower() + "." + val.lower() + "__"
                )

        for key in schema:
            id_map[key.lower()] = "__" + key.lower() + "__"

        return id_map


def get_schema(database_path: str) -> Dict[str, List[str]]:
    """Extract database schema from SQLite database.

    Args:
        database_path: Path to SQLite database file.

    Returns:
        Dictionary mapping table names to lists of column names.
    """
    schema: Dict[str, List[str]] = {}
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


def get_schema_from_json(filepath: str) -> Dict[str, List[str]]:
    """Load database schema from JSON file.

    Args:
        filepath: Path to JSON schema file.

    Returns:
        Dictionary mapping table names to lists of column names.
    """
    with open(filepath) as f:
        data = json.load(f)

    schema: Dict[str, List[str]] = {}
    for entry in data:
        table = str(entry["table"].lower())
        cols = [str(col["column_name"].lower()) for col in entry["col_data"]]
        schema[table] = cols

    return schema


def tokenize(query_string: str) -> List[str]:
    """Tokenize SQL query into individual tokens.

    Handles string literals within double quotes and complex operators.

    Args:
        query_string: SQL query string.

    Returns:
        List of tokens.
    """
    query_string = str(query_string)
    query_string = query_string.replace("'", '"')
    quote_idxs = [idx for idx, char in enumerate(query_string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    vals: Dict[str, str] = {}
    for i in range(len(quote_idxs) - 1, -1, -2):
        qidx1 = quote_idxs[i - 1]
        qidx2 = quote_idxs[i]
        val = query_string[qidx1 : qidx2 + 1]
        key = f"__val_{qidx1}_{qidx2}__"
        query_string = query_string[:qidx1] + key + query_string[qidx2 + 1 :]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(query_string)]
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ("!", ">", "<")
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx - 1]
        if pre_tok in prefix:
            toks = toks[: eq_idx - 1] + [pre_tok + "="] + toks[eq_idx + 1 :]

    return toks


def scan_alias(tokens: List[str]) -> Dict[str, str]:
    """Scan tokens for table aliases.

    Args:
        tokens: List of SQL tokens.

    Returns:
        Dictionary mapping alias names to original table names.
    """
    as_idxs = [idx for idx, tok in enumerate(tokens) if tok == "as"]
    alias: Dict[str, str] = {}
    for idx in as_idxs:
        alias[tokens[idx + 1]] = tokens[idx - 1]
    return alias


def get_tables_with_alias(
    schema: Dict[str, List[str]], tokens: List[str]
) -> Dict[str, str]:
    """Map all table names including aliases to original table names.

    Args:
        schema: Database schema dictionary.
        tokens: Tokenized SQL query.

    Returns:
        Dictionary mapping table/alias names to original table names.
    """
    tables = scan_alias(tokens)
    for key in schema:
        assert (
            key not in tables
        ), f"Alias {key} has the same name in table"
        tables[key] = key
    return tables


def parse_col(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema_obj: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, str]:
    """Parse a column reference and return its unique identifier.

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.
        tables_with_alias: Mapping of table aliases to original names.
        schema_obj: Schema object for ID mapping.
        default_tables: List of available tables for unqualified columns.

    Returns:
        Tuple of (next_index, column_id).
    """
    tok = tokens[start_idx]
    if tok == "*":
        return start_idx + 1, schema_obj.idMap[tok]

    if "." in tok:
        alias, col = tok.split(".")
        key = tables_with_alias[alias] + "." + col
        return start_idx + 1, schema_obj.idMap[key]

    assert (
        default_tables is not None and len(default_tables) > 0
    ), "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema_obj.schema[table]:
            key = table + "." + tok
            return start_idx + 1, schema_obj.idMap[key]

    raise AssertionError(f"Error col: {tok}")


def parse_col_unit(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema_obj: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Tuple[int, str, bool]]:
    """Parse a column unit with optional aggregation.

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.
        tables_with_alias: Mapping of table aliases.
        schema_obj: Schema object for ID mapping.
        default_tables: Available tables.

    Returns:
        Tuple of (next_index, (agg_id, col_id, is_distinct)).
    """
    idx = start_idx
    len_ = len(tokens)
    is_block = False
    is_distinct = False
    if tokens[idx] == "(":
        is_block = True
        idx += 1

    if tokens[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(tokens[idx])
        idx += 1
        assert idx < len_ and tokens[idx] == "("
        idx += 1
        if tokens[idx] == "distinct":
            idx += 1
            is_distinct = True
        idx, col_id = parse_col(
            tokens, idx, tables_with_alias, schema_obj, default_tables
        )
        assert idx < len_ and tokens[idx] == ")"
        idx += 1
        return idx, (agg_id, col_id, is_distinct)

    if tokens[idx] == "distinct":
        idx += 1
        is_distinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(
        tokens, idx, tables_with_alias, schema_obj, default_tables
    )

    if is_block:
        assert tokens[idx] == ")"
        idx += 1

    return idx, (agg_id, col_id, is_distinct)


def parse_val_unit(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema_obj: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Tuple[int, str, Union[str, None]]]:
    """Parse a value unit (possibly with arithmetic operations).

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.
        tables_with_alias: Mapping of table aliases.
        schema_obj: Schema object.
        default_tables: Available tables.

    Returns:
        Tuple of (next_index, (unit_op_id, col_unit1, col_unit2)).
    """
    idx = start_idx
    len_ = len(tokens)
    is_block = False
    if tokens[idx] == "(":
        is_block = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index("none")

    idx, col_unit1 = parse_col_unit(
        tokens, idx, tables_with_alias, schema_obj, default_tables
    )
    if idx < len_ and tokens[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(tokens[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(
            tokens, idx, tables_with_alias, schema_obj, default_tables
        )

    if is_block:
        assert tokens[idx] == ")"
        idx += 1

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema_obj: Schema,
) -> Tuple[int, str, str]:
    """Parse a table unit reference.

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.
        tables_with_alias: Mapping of table aliases.
        schema_obj: Schema object.

    Returns:
        Tuple of (next_index, table_id, table_name).
    """
    idx = start_idx
    len_ = len(tokens)
    key = tables_with_alias[tokens[idx]]

    if idx + 1 < len_ and tokens[idx + 1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema_obj.idMap[key], key


def parse_value(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema_obj: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Union[float, str, Dict[str, Any]]]:
    """Parse a value (number, string, column unit, or subquery).

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.
        tables_with_alias: Mapping of table aliases.
        schema_obj: Schema object.
        default_tables: Available tables.

    Returns:
        Tuple of (next_index, value).
    """
    idx = start_idx
    len_ = len(tokens)

    is_block = False
    if tokens[idx] == "(":
        is_block = True
        idx += 1

    if tokens[idx] == "select":
        idx, val = parse_sql(tokens, idx, tables_with_alias, schema_obj)
    elif '"' in tokens[idx]:
        val = tokens[idx]
        idx += 1
    else:
        try:
            val = float(tokens[idx])
            idx += 1
        except ValueError:
            end_idx = idx
            while (
                end_idx < len_
                and tokens[end_idx] != ","
                and tokens[end_idx] != ")"
                and tokens[end_idx] != "and"
                and tokens[end_idx] not in CLAUSE_KEYWORDS
                and tokens[end_idx] not in JOIN_KEYWORDS
            ):
                end_idx += 1

            idx, val = parse_col_unit(
                tokens[start_idx:end_idx],
                0,
                tables_with_alias,
                schema_obj,
                default_tables,
            )
            idx = end_idx

    if is_block:
        assert tokens[idx] == ")"
        idx += 1

    return idx, val


def parse_condition(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema_obj: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, List[Union[Tuple[bool, int, Any, Any, Optional[Any]], str]]]:
    """Parse WHERE/HAVING condition clauses.

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.
        tables_with_alias: Mapping of table aliases.
        schema_obj: Schema object.
        default_tables: Available tables.

    Returns:
        Tuple of (next_index, conditions_list).
    """
    idx = start_idx
    len_ = len(tokens)
    conds: List[Union[Tuple[bool, int, Any, Any, Optional[Any]], str]] = []

    while idx < len_:
        idx, val_unit = parse_val_unit(
            tokens, idx, tables_with_alias, schema_obj, default_tables
        )
        not_op = False
        if tokens[idx] == "not":
            not_op = True
            idx += 1

        assert (
            idx < len_ and tokens[idx] in WHERE_OPS
        ), f"Error condition: idx: {idx}, tok: {tokens[idx]}"
        op_id = WHERE_OPS.index(tokens[idx])
        idx += 1
        val1 = val2 = None
        if op_id == WHERE_OPS.index("between"):
            idx, val1 = parse_value(
                tokens, idx, tables_with_alias, schema_obj, default_tables
            )
            assert tokens[idx] == "and"
            idx += 1
            idx, val2 = parse_value(
                tokens, idx, tables_with_alias, schema_obj, default_tables
            )
        else:
            idx, val1 = parse_value(
                tokens, idx, tables_with_alias, schema_obj, default_tables
            )
            val2 = None

        conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and (
            tokens[idx] in CLAUSE_KEYWORDS
            or tokens[idx] in (")", ";")
            or tokens[idx] in JOIN_KEYWORDS
        ):
            break

        if idx < len_ and tokens[idx] in COND_OPS:
            conds.append(tokens[idx])
            idx += 1

    return idx, conds


def parse_select(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema_obj: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Tuple[bool, List[Tuple[int, Any]]]]:
    """Parse SELECT clause.

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.
        tables_with_alias: Mapping of table aliases.
        schema_obj: Schema object.
        default_tables: Available tables.

    Returns:
        Tuple of (next_index, (is_distinct, val_units)).
    """
    idx = start_idx
    len_ = len(tokens)

    assert tokens[idx] == "select", "'select' not found"
    idx += 1
    is_distinct = False
    if idx < len_ and tokens[idx] == "distinct":
        idx += 1
        is_distinct = True
    val_units: List[Tuple[int, Any]] = []

    while idx < len_ and tokens[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if tokens[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(tokens[idx])
            idx += 1
        idx, val_unit = parse_val_unit(
            tokens, idx, tables_with_alias, schema_obj, default_tables
        )
        val_units.append((agg_id, val_unit))
        if idx < len_ and tokens[idx] == ",":
            idx += 1

    return idx, (is_distinct, val_units)


def parse_from(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema_obj: Schema,
) -> Tuple[int, List[Tuple[str, str]], List[Any], List[str]]:
    """Parse FROM clause with joins.

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.
        tables_with_alias: Mapping of table aliases.
        schema_obj: Schema object.

    Returns:
        Tuple of (next_index, table_units, conditions, default_tables).
    """
    assert "from" in tokens[start_idx:], "'from' not found"

    len_ = len(tokens)
    idx = tokens.index("from", start_idx) + 1
    default_tables: List[str] = []
    table_units: List[Tuple[str, str]] = []
    conds: List[Any] = []

    while idx < len_:
        is_block = False
        if tokens[idx] == "(":
            is_block = True
            idx += 1

        if tokens[idx] == "select":
            idx, sql = parse_sql(tokens, idx, tables_with_alias, schema_obj)
            table_units.append((TABLE_TYPE["sql"], sql))
        else:
            if idx < len_ and tokens[idx] == "join":
                idx += 1
            idx, table_unit, table_name = parse_table_unit(
                tokens, idx, tables_with_alias, schema_obj
            )
            table_units.append((TABLE_TYPE["table_unit"], table_unit))
            default_tables.append(table_name)
        if idx < len_ and tokens[idx] == "on":
            idx += 1
            idx, this_conds = parse_condition(
                tokens, idx, tables_with_alias, schema_obj, default_tables
            )
            if len(conds) > 0:
                conds.append("and")
            conds.extend(this_conds)

        if is_block:
            assert tokens[idx] == ")"
            idx += 1
        if idx < len_ and (
            tokens[idx] in CLAUSE_KEYWORDS or tokens[idx] in (")", ";")
        ):
            break

    return idx, table_units, conds, default_tables


def parse_where(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema_obj: Schema,
    default_tables: List[str],
) -> Tuple[int, List[Any]]:
    """Parse WHERE clause.

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.
        tables_with_alias: Mapping of table aliases.
        schema_obj: Schema object.
        default_tables: Available tables.

    Returns:
        Tuple of (next_index, conditions).
    """
    idx = start_idx
    len_ = len(tokens)

    if idx >= len_ or tokens[idx] != "where":
        return idx, []

    idx += 1
    idx, conds = parse_condition(tokens, idx, tables_with_alias, schema_obj, default_tables)
    return idx, conds


def parse_group_by(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema_obj: Schema,
    default_tables: List[str],
) -> Tuple[int, List[Tuple[int, str, bool]]]:
    """Parse GROUP BY clause.

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.
        tables_with_alias: Mapping of table aliases.
        schema_obj: Schema object.
        default_tables: Available tables.

    Returns:
        Tuple of (next_index, col_units).
    """
    idx = start_idx
    len_ = len(tokens)
    col_units: List[Tuple[int, str, bool]] = []

    if idx >= len_ or tokens[idx] != "group":
        return idx, col_units

    idx += 1
    assert tokens[idx] == "by"
    idx += 1

    while idx < len_ and not (
        tokens[idx] in CLAUSE_KEYWORDS or tokens[idx] in (")", ";")
    ):
        idx, col_unit = parse_col_unit(
            tokens, idx, tables_with_alias, schema_obj, default_tables
        )
        col_units.append(col_unit)
        if idx < len_ and tokens[idx] == ",":
            idx += 1
        else:
            break

    return idx, col_units


def parse_order_by(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema_obj: Schema,
    default_tables: List[str],
) -> Tuple[int, Tuple[str, List[Any]]]:
    """Parse ORDER BY clause.

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.
        tables_with_alias: Mapping of table aliases.
        schema_obj: Schema object.
        default_tables: Available tables.

    Returns:
        Tuple of (next_index, (order_type, val_units)).
    """
    idx = start_idx
    len_ = len(tokens)
    val_units: List[Any] = []
    order_type = "asc"

    if idx >= len_ or tokens[idx] != "order":
        return idx, (order_type, val_units)

    idx += 1
    assert tokens[idx] == "by"
    idx += 1

    while idx < len_ and not (
        tokens[idx] in CLAUSE_KEYWORDS or tokens[idx] in (")", ";")
    ):
        idx, val_unit = parse_val_unit(
            tokens, idx, tables_with_alias, schema_obj, default_tables
        )
        val_units.append(val_unit)
        if idx < len_ and tokens[idx] in ORDER_OPS:
            order_type = tokens[idx]
            idx += 1
        if idx < len_ and tokens[idx] == ",":
            idx += 1
        else:
            break

    return idx, (order_type, val_units)


def parse_having(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema_obj: Schema,
    default_tables: List[str],
) -> Tuple[int, List[Any]]:
    """Parse HAVING clause.

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.
        tables_with_alias: Mapping of table aliases.
        schema_obj: Schema object.
        default_tables: Available tables.

    Returns:
        Tuple of (next_index, conditions).
    """
    idx = start_idx
    len_ = len(tokens)

    if idx >= len_ or tokens[idx] != "having":
        return idx, []

    idx += 1
    idx, conds = parse_condition(tokens, idx, tables_with_alias, schema_obj, default_tables)
    return idx, conds


def parse_limit(tokens: List[str], start_idx: int) -> Tuple[int, Optional[int]]:
    """Parse LIMIT clause.

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.

    Returns:
        Tuple of (next_index, limit_value).
    """
    idx = start_idx
    len_ = len(tokens)

    if idx < len_ and tokens[idx] == "limit":
        idx += 2
        if type(tokens[idx - 1]) != int:
            return idx, 1

        return idx, int(tokens[idx - 1])

    return idx, None


def skip_semicolon(tokens: List[str], start_idx: int) -> int:
    """Skip semicolon tokens.

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.

    Returns:
        Next index after semicolons.
    """
    idx = start_idx
    while idx < len(tokens) and tokens[idx] == ";":
        idx += 1
    return idx


def parse_sql(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema_obj: Schema,
) -> Tuple[int, Dict[str, Any]]:
    """Parse complete SQL query into AST.

    Args:
        tokens: Tokenized SQL query.
        start_idx: Starting index in tokens.
        tables_with_alias: Mapping of table aliases.
        schema_obj: Schema object.

    Returns:
        Tuple of (next_index, sql_dict).
    """
    is_block = False
    len_ = len(tokens)
    idx = start_idx

    sql: Dict[str, Any] = {}
    if tokens[idx] == "(":
        is_block = True
        idx += 1

    from_end_idx, table_units, conds, default_tables = parse_from(
        tokens, start_idx, tables_with_alias, schema_obj
    )
    sql["from"] = {"table_units": table_units, "conds": conds}
    _, select_col_units = parse_select(
        tokens, idx, tables_with_alias, schema_obj, default_tables
    )
    idx = from_end_idx
    sql["select"] = select_col_units
    idx, where_conds = parse_where(
        tokens, idx, tables_with_alias, schema_obj, default_tables
    )
    sql["where"] = where_conds
    idx, group_col_units = parse_group_by(
        tokens, idx, tables_with_alias, schema_obj, default_tables
    )
    sql["groupBy"] = group_col_units
    idx, having_conds = parse_having(
        tokens, idx, tables_with_alias, schema_obj, default_tables
    )
    sql["having"] = having_conds
    idx, order_col_units = parse_order_by(
        tokens, idx, tables_with_alias, schema_obj, default_tables
    )
    sql["orderBy"] = order_col_units
    idx, limit_val = parse_limit(tokens, idx)
    sql["limit"] = limit_val

    idx = skip_semicolon(tokens, idx)
    if is_block:
        assert tokens[idx] == ")"
        idx += 1
    idx = skip_semicolon(tokens, idx)

    for op in SQL_OPS:
        sql[op] = None
    if idx < len_ and tokens[idx] in SQL_OPS:
        sql_op = tokens[idx]
        idx += 1
        idx, iue_sql = parse_sql(tokens, idx, tables_with_alias, schema_obj)
        sql[sql_op] = iue_sql
    return idx, sql


def load_data(filepath: str) -> List[Dict[str, Any]]:
    """Load JSON data from file.

    Args:
        filepath: Path to JSON file.

    Returns:
        List of loaded data entries.
    """
    with open(filepath) as f:
        data = json.load(f)
    return data


def get_sql(schema_obj: Schema, query: str) -> Dict[str, Any]:
    """Parse SQL query into AST structure.

    Args:
        schema_obj: Schema object for ID mapping.
        query: SQL query string.

    Returns:
        Parsed SQL as dictionary (AST).
    """
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema_obj.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema_obj)

    return sql
