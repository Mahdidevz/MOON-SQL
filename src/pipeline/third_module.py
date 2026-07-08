"""Third-round SQL correction via LLM-based reflection and repair."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
from typing import Any, Dict, List, Optional

import tiktoken
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from prompts import (
    correct_prompt,
    correct_prompt_kg,
    reflect_prompt,
    reflect_prompt_kg,
    sql_middle_prompt,
    sql_middle_prompt_kg,
)
from utils.tools import (
    generate_foreign_key,
    generate_foreign_key_by_tables,
    generate_schema_by_dict_only,
    generate_schema_simple,
    new_run_sql,
)

MODULE = "[third_round]"

input_length: int = 0
output_length: int = 0


def log(message: str) -> None:
    """Print a prefixed, flushed log line to stdout."""
    print(f"{MODULE} {message}", flush=True)


def parse_option() -> argparse.Namespace:
    """Parse and return CLI arguments."""
    parser = argparse.ArgumentParser("")
    parser.add_argument("--dev_path", type=str, default="dev.json")
    parser.add_argument("--data_path", type=str, default="preprocessed_data.json")
    parser.add_argument("--input_path", type=str, default="second_round.sql")
    parser.add_argument("--output_path", type=str, default="third_round.sql")
    parser.add_argument("--db_path", type=str, default="database")
    parser.add_argument("--retry_num", type=int, default=10)
    parser.add_argument("--process_num", type=int, default=1)
    parser.add_argument("--schema_path", type=str, default=None)
    parser.add_argument(
        "--short_model_name", type=str, default="llama-3.3-70b-versatile"
    )
    parser.add_argument(
        "--long_model_name", type=str, default="llama-3.3-70b-versatile"
    )
    return parser.parse_args()


class ReflectTool:
    """Generates a natural-language explanation of why a SQL query failed."""

    def __init__(
        self,
        short_model_name: str = "llama-3.3-70b-versatile",
        long_model_name: str = "llama-3.3-70b-versatile",
    ) -> None:
        """Initialise with short- and long-context LLM backends.

        Args:
            short_model_name: Model used for prompts under 3 800 tokens.
            long_model_name: Model used for prompts at or above 3 800 tokens.
        """
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.prompt_template_kg = reflect_prompt_kg
        self.prompt_template = reflect_prompt
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=short_model_name,  # type: ignore[call-arg]
            request_timeout=600,  # type: ignore[call-arg]
            max_retries=3,
        )
        self.llm_long = ChatOpenAI(
            temperature=0,
            model_name=long_model_name,  # type: ignore[call-arg]
            request_timeout=600,  # type: ignore[call-arg]
            max_retries=3,
        )

    def run(
        self,
        question: str,
        schema: str,
        foreign_keys: str,
        previous_information: str,
        knowledge: Optional[str] = None,
    ) -> str:
        """Return a reflection string describing the probable SQL error cause.

        Args:
            question: Natural-language question.
            schema: Database schema string.
            foreign_keys: Foreign-key relationship string.
            previous_information: Prior SQL attempts and error messages.
            knowledge: Optional external knowledge string.

        Returns:
            Reflection text from the LLM.
        """
        if knowledge is not None:
            prompt = self.prompt_template_kg.format(
                question=question,
                schema=schema,
                foreign_keys=foreign_keys,
                previous_information=previous_information,
                knowledge=knowledge,
            ).strip()
        else:
            prompt = self.prompt_template.format(
                question=question,
                schema=schema,
                foreign_keys=foreign_keys,
                previous_information=previous_information,
            ).strip()

        prompt = "\n".join([" ".join(e.split()) for e in prompt.split("\n")])

        if len(self.encoder.encode(prompt)) < 3800:
            log("Sending reflection prompt to OpenRouter API")
            return self.llm.predict(prompt)
        log("Sending long reflection prompt to OpenRouter API")
        return self.llm_long.predict(prompt)


class CorrectTool:
    """Generates a corrected SQL query given prior error context."""

    def __init__(
        self,
        short_model_name: str = "llama-3.3-70b-versatile",
        long_model_name: str = "llama-3.3-70b-versatile",
    ) -> None:
        """Initialise with short- and long-context LLM backends.

        Args:
            short_model_name: Model used for prompts under 3 800 tokens.
            long_model_name: Model used for prompts at or above 3 800 tokens.
        """
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.prompt_template_kg = correct_prompt_kg
        self.prompt_template = correct_prompt
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=short_model_name,  # type: ignore[call-arg]
            request_timeout=600,  # type: ignore[call-arg]
            max_retries=3,
        )
        self.llm_long = ChatOpenAI(
            temperature=0,
            model_name=long_model_name,  # type: ignore[call-arg]
            request_timeout=600,  # type: ignore[call-arg]
            max_retries=3,
        )

    def run(
        self,
        question: str,
        schema: str,
        foreign_keys: str,
        previous_information: str,
        knowledge: Optional[str] = None,
    ) -> str:
        """Return a corrected SQL query string.

        Args:
            question: Natural-language question.
            schema: Database schema string.
            foreign_keys: Foreign-key relationship string.
            previous_information: Prior SQL attempts and error messages.
            knowledge: Optional external knowledge string.

        Returns:
            Corrected SQL query string.
        """
        if knowledge is not None:
            prompt = self.prompt_template_kg.format(
                question=question,
                schema=schema,
                foreign_keys=foreign_keys,
                previous_information=previous_information,
                knowledge=knowledge,
            ).strip()
        else:
            prompt = self.prompt_template.format(
                question=question,
                schema=schema,
                foreign_keys=foreign_keys,
                previous_information=previous_information,
            ).strip()

        prompt = "\n".join([" ".join(e.split()) for e in prompt.split("\n")])

        if len(self.encoder.encode(prompt)) < 3800:
            log("Sending correction prompt to OpenRouter API")
            sql = "SELECT " + self.llm.predict(prompt)
        else:
            log("Sending long correction prompt to OpenRouter API")
            sql = "SELECT " + self.llm_long.predict(prompt)

        sql = (
            sql.replace("```sql\n", "")
            .replace("```\n", "")
            .replace("\n```", "")
            .replace("SELECTsql\n", "")
            .replace("```", "")
        )
        sql = sql.replace("SELECT SELECT", "SELECT")
        sql = sql.replace("\n", " ")
        sql = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")

        last_sql = copy.deepcopy(sql)
        while last_sql != sql:
            last_sql = copy.deepcopy(sql)
            sql = sql.replace("  ", " ")
        return sql


class SQLGenerateTool:
    """Generates a fresh SQL query from question and schema context."""

    def __init__(
        self,
        short_model_name: str = "llama-3.3-70b-versatile",
        long_model_name: str = "llama-3.3-70b-versatile",
    ) -> None:
        """Initialise with short- and long-context LLM backends.

        Args:
            short_model_name: Model used for prompts under 3 800 tokens.
            long_model_name: Model used for prompts at or above 3 800 tokens.
        """
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.prompt_template_kg = sql_middle_prompt_kg
        self.prompt_template = sql_middle_prompt
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=short_model_name,  # type: ignore[call-arg]
            request_timeout=600,  # type: ignore[call-arg]
            max_retries=3,
        )
        self.llm_long = ChatOpenAI(
            temperature=0.0,
            model_name=long_model_name,  # type: ignore[call-arg]
            request_timeout=600,  # type: ignore[call-arg]
            max_retries=3,
        )

    def run(
        self,
        question: str,
        schema: str,
        foreign_keys: str,
        knowledge: Optional[str] = None,
    ) -> str:
        """Generate a SQL query and log cumulative token usage to disk.

        Args:
            question: Natural-language question.
            schema: Database schema string.
            foreign_keys: Foreign-key relationship string.
            knowledge: Optional external knowledge string.

        Returns:
            Generated SQL query string.
        """
        global input_length, output_length

        if knowledge is not None:
            prompt = self.prompt_template_kg.format(
                question=question,
                schema=schema,
                foreign_keys=foreign_keys,
                knowledge=knowledge,
            ).strip()
        else:
            prompt = self.prompt_template.format(
                question=question,
                schema=schema,
                foreign_keys=foreign_keys,
            ).strip()

        prompt = "\n".join([" ".join(e.split()) for e in prompt.split("\n")])

        if len(self.encoder.encode(prompt)) < 3800:
            log("Sending SQL generation prompt to OpenRouter API")
            sql = "SELECT " + self.llm.predict(prompt)
        else:
            log("Sending long SQL generation prompt to OpenRouter API")
            sql = "SELECT " + self.llm_long.predict(prompt)

        input_length += len(self.encoder.encode(prompt))
        output_length += len(self.encoder.encode(sql))
        with open("third_round_input.txt", "a") as f:
            f.write(str(input_length) + "\n")
        with open("third_round_output.txt", "a") as f:
            f.write(str(output_length) + "\n")

        sql = sql.replace("SELECT SELECT", "SELECT")
        sql = sql.replace("\n", " ")
        sql = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
        return sql


def get_output_name(path: str, idx: int) -> str:
    """Return a chunk-specific variant of *path* with *idx* inserted before the extension."""
    parts = path.split(".")
    parts[-2] = parts[-2] + str(idx)
    return ".".join(parts)


def correct_sql(
    idx: int,
    start: int,
    end: int,
    dev: List[Dict[str, Any]],
    data_all: List[Dict[str, Any]],
    sqls: List[str],
    output_path: str,
    db_path: str,
    max_retry_num: int,
    schema_path: Optional[str],
    short_model_name: str,
    long_model_name: str,
) -> None:
    """Execute, reflect on, and correct SQL queries for a slice of the dataset.

    Args:
        idx: Worker index for output-file partitioning.
        start: First sample index (inclusive).
        end: Last sample index (exclusive).
        dev: Development dataset records.
        data_all: Preprocessed schema records.
        sqls: Input SQL predictions to validate and correct.
        output_path: Base path for the per-worker output file.
        db_path: Root directory of SQLite database files.
        max_retry_num: Maximum correction attempts per failing query.
        schema_path: Optional path to a pre-computed schema selection file.
        short_model_name: Model for prompts under 3 800 tokens.
        long_model_name: Model for prompts at or above 3 800 tokens.
    """
    reflector = ReflectTool(short_model_name, long_model_name)
    corrector = CorrectTool(short_model_name, long_model_name)
    generator = SQLGenerateTool(short_model_name, long_model_name)  # noqa: F841

    temp_output_path = get_output_name(output_path, idx)
    if os.path.exists(temp_output_path):
        with open(temp_output_path, "r") as f:
            start = len(f.readlines()) + start

    all_schema: Optional[List[Dict[str, Any]]] = None
    if schema_path is not None:
        with open(schema_path) as f:
            all_schema = json.load(f)

    error_num = 0
    error_not_solve = 0

    with open(temp_output_path, "a+") as f_tmp:
        log(f"Worker {idx + 1}: processing samples {start + 1}-{end} of {len(dev)}")
        for i in tqdm(range(len(sqls))[start:end]):
            log(f"Sample {i + 1}/{len(sqls)}: executing and correcting SQL")
            db_id = dev[i]["db_id"]
            db_dir = f"{db_path}/{db_id}/{db_id}.sqlite"
            sql = sqls[i].strip()

            result, flag = new_run_sql(db_dir, sql)
            times = 0
            last_sqls = [sql]
            last_errors = [str(result)]

            question = dev[i]["question"]
            knowledge = dev[i].get("evidence")

            if all_schema is None:
                foreign_keys = generate_foreign_key(data_all[i])
                schema = generate_schema_simple(data_all[i])
            else:
                foreign_keys = generate_foreign_key_by_tables(
                    data_all[i], list(all_schema[i].keys())
                )
                schema = generate_schema_by_dict_only(data_all[i], all_schema[i])

            previous_information = (
                f"### SQL: {last_sqls[-1]}\n### Error message: {last_errors[-1]}"
            )

            if flag is False:
                error_num += 1

            try:
                while flag is False:
                    sql_reason = reflector.run(
                        question, schema, foreign_keys, previous_information, knowledge
                    )
                    previous_information += f"\n### Error Reason: {sql_reason}"
                    sql = corrector.run(
                        question, schema, foreign_keys, previous_information, knowledge
                    )
                    result, flag = new_run_sql(db_dir, sql)
                    previous_information += (
                        f"\n### new SQL: {sql}\n### Error message: {result}"
                    )
                    last_sqls.append(sql)
                    last_errors.append(str(result))
                    times += 1
                    if times >= max_retry_num:
                        error_not_solve += 1
                        break
            except Exception:
                sql = sqls[i].strip()

            f_tmp.write(sql + "\n")
            f_tmp.flush()

    log(f"Worker {idx + 1}: finished chunk; saved {temp_output_path}")


def correct_parallel(
    dev: List[Dict[str, Any]],
    data_all: List[Dict[str, Any]],
    sqls: List[str],
    output_path: str,
    db_path: str,
    max_retry_num: int,
    schema_path: Optional[str],
    short_model_name: str,
    long_model_name: str,
    process_num: int,
) -> List[Any]:
    """Distribute SQL correction across a thread pool and return worker results."""
    contents: List[Any] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                correct_sql,
                i,
                i * len(dev) // process_num,
                min((i + 1) * len(dev) // process_num, len(dev)),
                dev,
                data_all,
                sqls,
                output_path,
                db_path,
                max_retry_num,
                schema_path,
                short_model_name,
                long_model_name,
            )
            for i in range(process_num)
        ]
        for future in concurrent.futures.as_completed(futures):
            contents.append(future.result())
    return contents


def main(opt: argparse.Namespace) -> None:
    """Correct SQL for all samples in parallel, then merge chunk files into one output."""
    log(f"Starting: input_path={opt.input_path}, output_path={opt.output_path}")

    with open(opt.dev_path) as f:
        dev: List[Dict[str, Any]] = json.load(f)
    with open(opt.data_path) as f:
        data_all: List[Dict[str, Any]] = json.load(f)
    with open(opt.input_path, "r") as f:
        last_sqls = [line.strip() for line in f.readlines()]

    if os.path.exists(opt.output_path):
        log(f"Output {opt.output_path} already exists, skipping")
        return

    correct_parallel(
        dev,
        data_all,
        last_sqls,
        opt.output_path,
        opt.db_path,
        opt.retry_num,
        opt.schema_path,
        opt.short_model_name,
        opt.long_model_name,
        opt.process_num,
    )

    log(f"Merging {opt.process_num} chunk files into {opt.output_path}")
    with open(opt.output_path, "w") as f_out:
        for i in range(opt.process_num):
            chunk_path = get_output_name(opt.output_path, i)
            with open(chunk_path, "r") as f_chunk:
                data = f_chunk.readlines()
            os.remove(chunk_path)
            for pre_sql in data:
                f_out.write(pre_sql.strip("\n") + "\n")

    log(f"Finished; saved SQL to {opt.output_path}")


if __name__ == "__main__":
    opt = parse_option()
    main(opt)
