"""Utilities for fetching and matching database column values against question tokens.

This module provides BM25-assisted and fuzzy-match-based retrieval of column
picklists from SQLite databases, used during question pre-processing to
surface relevant cell values for NL-to-SQL models.
"""

import difflib
import os
import pickle
import sqlite3
from typing import List, Optional, Tuple, Union

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

# fmt: off
_stopwords = {'who', 'ourselves', 'down', 'only', 'were', 'him', 'at', "weren't", 'has', 'few', "it's", 'm', 'again',
              'd', 'haven', 'been', 'other', 'we', 'an', 'own', 'doing', 'ma', 'hers', 'all', "haven't", 'in', 'but',
              "shouldn't", 'does', 'out', 'aren', 'you', "you'd", 'himself', "isn't", 'most', 'y', 'below', 'is',
              "wasn't", 'hasn', 'them', 'wouldn', 'against', 'this', 'about', 'there', 'don', "that'll", 'a', 'being',
              'with', 'your', 'theirs', 'its', 'any', 'why', 'now', 'during', 'weren', 'if', 'should', 'those', 'be',
              'they', 'o', 't', 'of', 'or', 'me', 'i', 'some', 'her', 'do', 'will', 'yours', 'for', 'mightn', 'nor',
              'needn', 'the', 'until', "couldn't", 'he', 'which', 'yourself', 'to', "needn't", "you're", 'because',
              'their', 'where', 'it', "didn't", 've', 'whom', "should've", 'can', "shan't", 'on', 'had', 'have',
              'myself', 'am', "don't", 'under', 'was', "won't", 'these', 'so', 'as', 'after', 'above', 'each', 'ours',
              'hadn', 'having', 'wasn', 's', 'doesn', "hadn't", 'than', 'by', 'that', 'both', 'herself', 'his',
              "wouldn't", 'into', "doesn't", 'before', 'my', 'won', 'more', 'are', 'through', 'same', 'how', 'what',
              'over', 'll', 'yourselves', 'up', 'mustn', "mustn't", "she's", 're', 'such', 'didn', "you'll", 'shan',
              'when', "you've", 'themselves', "mightn't", 'she', 'from', 'isn', 'ain', 'between', 'once', 'here',
              'shouldn', 'our', 'and', 'not', 'too', 'very', 'further', 'while', 'off', 'couldn', "hasn't", 'itself',
              'then', 'did', 'just', "aren't"}
# fmt: on

_commonwords = {"no", "yes", "many"}


def is_number(s: str) -> bool:
    """Return True if *s* can be parsed as a float after stripping commas.

    Args:
        s: The string to test.

    Returns:
        True when ``float(s.replace(",", ""))`` succeeds, False otherwise.
    """
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def is_stopword(s: str) -> bool:
    """Return True if the stripped form of *s* is an English stopword.

    Args:
        s: The string to test.

    Returns:
        True when ``s.strip()`` is found in the module-level ``_stopwords`` set.
    """
    return s.strip() in _stopwords


def is_commonword(s: str) -> bool:
    """Return True if the stripped form of *s* is a common word (no/yes/many).

    Args:
        s: The string to test.

    Returns:
        True when ``s.strip()`` is found in the module-level ``_commonwords`` set.
    """
    return s.strip() in _commonwords


def is_common_db_term(s: str) -> bool:
    """Return True if *s* is a ubiquitous database term (e.g. ``id``).

    Args:
        s: The string to test.

    Returns:
        True when ``s.strip()`` equals ``"id"``.
    """
    return s.strip() in ["id"]


class Match:
    """Lightweight value object describing a substring match position.

    Attributes:
        start: Zero-based start index of the match within the source string.
        size: Length of the matched substring in characters.
    """

    def __init__(self, start: int, size: int) -> None:
        """Initialise a Match with a start position and length.

        Args:
            start: Zero-based start index of the match.
            size: Length of the matched region.
        """
        self.start = start
        self.size = size


def is_span_separator(c: str) -> bool:
    """Return True if *c* is a character that separates token spans.

    Args:
        c: A single character.

    Returns:
        True when *c* is one of the characters ``'"()`,.?!`` or a space.
    """
    return c in "'\"()`,.?! "


def split(s: str) -> List[str]:
    """Tokenise *s* into a list of individual lower-cased characters.

    Args:
        s: The string to split.

    Returns:
        A list of single lower-cased characters from the stripped input.
    """
    return [c.lower() for c in s.strip()]


def prefix_match(s1: str, s2: str) -> bool:
    """Return True if the first non-separator characters of *s1* and *s2* match.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        True when both strings share the same first non-separator character,
        or when both strings consist entirely of separators.
    """
    i, j = 0, 0
    for i in range(len(s1)):
        if not is_span_separator(s1[i]):
            break
    for j in range(len(s2)):
        if not is_span_separator(s2[j]):
            break
    if i < len(s1) and j < len(s2):
        return s1[i] == s2[j]
    elif i >= len(s1) and j >= len(s2):
        return True
    else:
        return False


def get_effective_match_source(
    s: Union[str, List[str]], start: int, end: int
) -> Optional[Match]:
    """Find the effective token-level match region within *s* around [start, end).

    Expands the raw character-level match boundaries outward to the nearest
    span-separator characters, then trims leading/trailing separators to
    return the tightest enclosing non-separator span.

    Args:
        s: The source character list (typically produced by :func:`split`).
        start: Start index of the raw match in *s*.
        end: End index (exclusive) of the raw match in *s*.

    Returns:
        A :class:`Match` describing the effective region, or ``None`` if a
        valid region cannot be determined.
    """
    _start = -1

    for i in range(start, start - 2, -1):
        if i < 0:
            _start = i + 1
            break
        if is_span_separator(s[i]):
            _start = i
            break

    if _start < 0:
        return None

    _end = -1
    for i in range(end - 1, end + 3):
        if i >= len(s):
            _end = i - 1
            break
        if is_span_separator(s[i]):
            _end = i
            break

    if _end < 0:
        return None

    while _start < len(s) and is_span_separator(s[_start]):
        _start += 1
    while _end >= 0 and is_span_separator(s[_end]):
        _end -= 1

    return Match(_start, _end - _start + 1)


def get_matched_entries(
    s: str,
    field_values: List[str],
    m_theta: float = 0.85,
    s_theta: float = 0.85,
) -> Optional[List[Tuple[str, Tuple[str, str, float, float, int]]]]:
    """Find field values whose longest common substring with *s* meets score thresholds.

    Uses :mod:`difflib` for sequence matching and :mod:`rapidfuzz` for fuzzy
    ratio scoring. Results are sorted by a composite score
    ``(1e16 * match_score + 1e8 * s_match_score + match_size)`` in descending
    order.

    Args:
        s: The question string (or pre-split character list) to match against.
        field_values: Candidate cell values to search within.
        m_theta: Minimum match score threshold (0–1).
        s_theta: Minimum source-match score threshold (0–1).

    Returns:
        A sorted list of ``(match_str, (field_value, source_match_str,
        match_score, s_match_score, match_size))`` tuples, or ``None`` when
        no entry clears the thresholds.
    """
    if not field_values:
        return None

    if isinstance(s, str):
        n_grams = split(s)
    else:
        n_grams = s

    matched = dict()
    for field_value in field_values:
        if not isinstance(field_value, str):
            continue
        fv_tokens = split(field_value)
        sm = difflib.SequenceMatcher(None, n_grams, fv_tokens)
        match = sm.find_longest_match(0, len(n_grams), 0, len(fv_tokens))
        if match.size > 0:
            source_match = get_effective_match_source(
                n_grams, match.a, match.a + match.size
            )
            if source_match and source_match.size > 1:
                match_str = field_value[match.b : match.b + match.size]
                source_match_str = s[
                    source_match.start : source_match.start + source_match.size
                ]
                c_match_str = match_str.lower().strip()
                c_source_match_str = source_match_str.lower().strip()
                c_field_value = field_value.lower().strip()
                if (
                    c_match_str
                    and not is_number(c_match_str)
                    and not is_common_db_term(c_match_str)
                ):
                    if (
                        is_stopword(c_match_str)
                        or is_stopword(c_source_match_str)
                        or is_stopword(c_field_value)
                    ):
                        continue
                    if c_source_match_str.endswith(c_match_str + "'s"):
                        match_score = 1.0
                    else:
                        if prefix_match(c_field_value, c_source_match_str):
                            match_score = (
                                fuzz.ratio(c_field_value, c_source_match_str) / 100
                            )
                        else:
                            match_score = 0
                    if (
                        is_commonword(c_match_str)
                        or is_commonword(c_source_match_str)
                        or is_commonword(c_field_value)
                    ) and match_score < 1:
                        continue
                    s_match_score = match_score
                    if match_score >= m_theta and s_match_score >= s_theta:
                        if field_value.isupper() and match_score * s_match_score < 1:
                            continue
                        matched[match_str] = (
                            field_value,
                            source_match_str,
                            match_score,
                            s_match_score,
                            match.size,
                        )

    if not matched:
        return None
    else:
        return sorted(
            matched.items(),
            key=lambda x: 1e16 * x[1][2] + 1e8 * x[1][3] + x[1][4],
            reverse=True,
        )


def get_column_picklist(table_name: str, column_name: str, db_path: str) -> list:
    """Return the list of distinct values for *column_name* in *table_name*.

    Connects to the SQLite database at *db_path*, fetches all distinct values
    for the specified column, and normalises byte strings to UTF-8 (falling
    back to latin-1 on decode errors). Empty strings and ``None`` values are
    filtered out.

    Args:
        table_name: Name of the SQLite table to query.
        column_name: Name of the column whose distinct values are fetched.
        db_path: File-system path to the ``.sqlite`` database file.

    Returns:
        A list of distinct column values with empty/null entries removed.
    """
    # fetch_sql = "SELECT DISTINCT `{}` FROM `{}` LIMIT 100;".format(column_name, table_name)
    fetch_sql = "SELECT DISTINCT `{}` FROM `{}`;".format(column_name, table_name)
    conn = sqlite3.connect(db_path)
    conn.text_factory = bytes
    try:
        c = conn.cursor()
        c.execute(fetch_sql)
        picklist = set()
        for x in c.fetchall():
            if isinstance(x[0], str):
                picklist.add(x[0].encode("utf-8"))
            elif isinstance(x[0], bytes):
                try:
                    picklist.add(x[0].decode("utf-8"))
                except UnicodeDecodeError:
                    picklist.add(x[0].decode("latin-1"))
            else:
                picklist.add(x[0])
        picklist = list(picklist)
        picklist = [e for e in picklist if str(e) != "" and e is not None]
        # picklist = sorted(picklist)
    except Exception:
        conn.text_factory = bytes
        c = conn.cursor()
        c.execute(fetch_sql)
        picklist = set()
        for x in c.fetchall():
            if isinstance(x[0], str):
                picklist.add(x[0].encode("utf-8"))
            elif isinstance(x[0], bytes):
                try:
                    picklist.add(x[0].decode("utf-8"))
                except UnicodeDecodeError:
                    picklist.add(x[0].decode("latin-1"))
            else:
                picklist.add(x[0])
        picklist = list(picklist)
        picklist = [e for e in picklist if str(e) != "" and e is not None]
    finally:
        conn.close()
    return picklist


def get_db_content_more(
    table_name: str,
    column_name: str,
    db_path: str,
    k: int = 3,
) -> List[str]:
    """Return the first *k* distinct values from *column_name* in *table_name*.

    Args:
        table_name: Name of the SQLite table to query.
        column_name: Name of the column whose values are retrieved.
        db_path: File-system path to the ``.sqlite`` database file.
        k: Maximum number of values to return. Defaults to 3.

    Returns:
        A list of up to *k* distinct column values.
    """
    pick_list = get_column_picklist(table_name, column_name, db_path)
    # pick_list = [e for e in pick_list if str(e) != '']
    return pick_list[:k]


def get_database_matches_with_bm25(
    question: str,
    table_name: str,
    column_name: str,
    db_path: str,
    db_id: str,
    top_k_matches: int = 2,
    match_threshold: float = 0.85,
    tmp_file: str = "temp_column_corpus_path",
) -> List[str]:
    """Retrieve cell values that best match *question* using BM25 pre-filtering.

    Corpus and BM25 model files are cached on disk under *tmp_file* so that
    subsequent calls for the same column avoid redundant computation. The BM25
    top-1000 candidates are then refined with :func:`get_matched_entries` for
    fuzzy matching.

    Args:
        question: The natural-language question string.
        table_name: Name of the SQLite table.
        column_name: Name of the column to search.
        db_path: File-system path to the ``.sqlite`` database file.
        db_id: Database identifier used to namespace the cache directory.
        top_k_matches: Maximum number of matching values to return.
        match_threshold: Fuzzy-match score threshold forwarded to
            :func:`get_matched_entries` (applied to both *m_theta* and
            *s_theta*).
        tmp_file: Root directory for on-disk corpus/BM25 caches.

    Returns:
        A list of up to *top_k_matches* cell values ranked by relevance.
    """
    if not os.path.exists(
        os.path.join(tmp_file, db_id, table_name, column_name, "corpus.pkl")
    ):
        os.makedirs(
            os.path.join(tmp_file, db_id, table_name, column_name), exist_ok=True
        )
        corpus = get_column_picklist(
            table_name=table_name, column_name=column_name, db_path=db_path
        )
        with open(
            os.path.join(tmp_file, db_id, table_name, column_name, "corpus.pkl"), "wb"
        ) as f:
            pickle.dump(corpus, f)
    else:
        with open(
            os.path.join(tmp_file, db_id, table_name, column_name, "corpus.pkl"), "rb"
        ) as f:
            corpus = pickle.load(f)
    # return corpus[:top_k_matches]
    if len(corpus) > 0:
        if not os.path.exists(
            os.path.join(tmp_file, db_id, table_name, column_name, "bm25.pkl")
        ):
            tokenized_corpus = [str(doc).split(" ") for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            with open(
                os.path.join(tmp_file, db_id, table_name, column_name, "bm25.pkl"), "wb"
            ) as f:
                pickle.dump(bm25, f)
        else:
            with open(
                os.path.join(tmp_file, db_id, table_name, column_name, "bm25.pkl"), "rb"
            ) as f:
                bm25 = pickle.load(f)
        tokenized_query = question.split(" ")
        picklist = bm25.get_top_n(tokenized_query, corpus, n=1000)
        matches = []

        if picklist and isinstance(picklist[0], str):
            matched_entries = get_matched_entries(
                s=question,
                field_values=picklist,
                m_theta=match_threshold,
                s_theta=match_threshold,
            )

            if matched_entries:
                num_values_inserted = 0
                for _match_str, (
                    field_value,
                    _s_match_str,
                    match_score,
                    s_match_score,
                    _match_size,
                ) in matched_entries:
                    if "name" in column_name and match_score * s_match_score < 1:
                        continue
                    if table_name != "sqlite_sequence":  # Spider database artifact
                        matches.append(field_value.strip())
                        num_values_inserted += 1
                        if num_values_inserted >= top_k_matches:
                            break
        for v in picklist:
            if len(matches) >= top_k_matches:
                break
            elif v not in matches:
                matches.append(v)
    else:
        matches = corpus[:top_k_matches]
    return matches
