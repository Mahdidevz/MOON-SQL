"""First-round SQL generation via LLM API with parallel thread processing."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from typing import Any, Dict, List, Optional

import tiktoken
from langchain_community.chat_models import ChatOpenAI
from tqdm import tqdm

from prompts import sql_simple_prompt, sql_simple_prompt_kg
from utils.tools import generate_foreign_key, generate_schema

MODULE = "[first_round]"


def log(message: str) -> None:
    """Print a prefixed, flushed log line to stdout."""
    print(f"{MODULE} {message}", flush=True)


def parse_option() -> argparse.Namespace:
    """Parse and return CLI arguments."""
    parser = argparse.ArgumentParser("First round SQL generation")
    parser.add_argument("--dev_path", type=str, default="dev.json")
    parser.add_argument(
        "--short_model_name", type=str, default="llama-3.3-70b-versatile"
    )
    parser.add_argument(
        "--long_model_name", type=str, default="llama-3.3-70b-versatile"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../generate_datasets_bird/preprocessed_data.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../intermediate_datasets_bird/first_round.sql",
    )
    parser.add_argument("--process_num", type=int, default=1)
    return parser.parse_args()


class SQLGenerateTool:
    """SQL generator that routes prompts to short- or long-context LLM models."""

    def __init__(
        self,
        short_model_name: str = "llama-3.3-70b-versatile",
        long_model_name: str = "llama-3.3-70b-versatile",
    ) -> None:
        """Initialise with two LLM backends selected by prompt token length.

        Args:
            short_model_name: Model used for prompts under 3 800 tokens.
            long_model_name: Model used for prompts at or above 3 800 tokens.
        """
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.prompt_template_kg = sql_simple_prompt_kg
        self.prompt_template = sql_simple_prompt
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=short_model_name,  # type: ignore[call-arg]
            request_timeout=60,  # type: ignore[call-arg]
            max_retries=10,
        )
        self.llm_long = ChatOpenAI(
            temperature=0,
            model_name=long_model_name,  # type: ignore[call-arg]
            request_timeout=60,  # type: ignore[call-arg]
            max_retries=10,
        )

    def run(
        self,
        question: str,
        schema: str,
        foreign_keys: str,
        knowledge: Optional[str] = None,
    ) -> str:
        """Generate a SQL query for the given question and schema context.

        Args:
            question: Natural-language question.
            schema: Database schema string.
            foreign_keys: Foreign-key relationship string.
            knowledge: Optional external knowledge string.

        Returns:
            Generated SQL query string.
        """
        if knowledge is not None:
            prompt = self.prompt_template_kg.format(
                question=question,
                schema=schema,
                foreign_keys=foreign_keys,
                knowledge=knowledge,
            ).strip()
        else:
            prompt = self.prompt_template.format(
                question=question, schema=schema, foreign_keys=foreign_keys
            ).strip()

        prompt = "\n".join([" ".join(e.split()) for e in prompt.split("\n")])
        sql = ""
        while sql == "":
            try:
                if len(self.encoder.encode(prompt)) < 3800:
                    sql = "SELECT  " + self.llm.predict(prompt)
                else:
                    sql = "SELECT  " + self.llm_long.predict(prompt)
            except Exception as e:
                log(f"Error during LLM call: {e}")

        sql = sql.replace("```sql\n", "")
        sql = sql.replace(" ```\n", "")
        sql = sql.replace("```", "")
        sql = sql.replace("SELECT  SELECT", "SELECT ")
        sql = sql.replace("SELECT SELECT", "SELECT ")
        sql = sql.replace("\n", " ")
        sql = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
        sql = sql.replace("SELECT  ", "SELECT ")
        return sql


def get_output_name(path: str, index: int) -> str:
    """Return a chunk-specific variant of *path* with *index* inserted before the extension."""
    parts = path.split(".")
    parts[-2] = parts[-2] + str(index)
    return ".".join(parts)


def generate_sql(
    start: int,
    end: int,
    ids: range,
    dev: List[Dict[str, Any]],
    idx: int,
    output_path: str,
    data_all: List[Dict[str, Any]],
    short_model_name: str,
    long_model_name: str,
) -> int:
    """Generate SQL for a contiguous slice of the dataset and append results to a file.

    Returns:
        1 on success.
    """
    sql_generator = SQLGenerateTool(short_model_name, long_model_name)
    temp_output_path = get_output_name(output_path, idx)

    if os.path.exists(temp_output_path):
        with open(temp_output_path, "r") as f:
            start = len(f.readlines()) + start

    with open(temp_output_path, "a+") as f_tmp:
        log(f"Worker {idx + 1}: processing samples {start + 1}-{end} of {len(dev)}")
        for use_id in tqdm(ids[start:end]):
            question = dev[use_id]["question"]
            knowledge = dev[use_id].get("evidence")
            foreign_keys = generate_foreign_key(data_all[use_id])
            schema = generate_schema(data_all[use_id])
            log(f"Sample {use_id + 1}/{len(dev)}: sending prompt to Groq API")
            pre_sql = sql_generator.run(question, schema, foreign_keys, knowledge)
            f_tmp.write(pre_sql + "\n")
            f_tmp.flush()
            log(f"Sample {use_id + 1}/{len(dev)}: saved SQL to {temp_output_path}")

    log(f"Worker {idx + 1}: finished chunk, saved to {temp_output_path}")
    return 1


def generate_parallel(
    ids: range,
    dev: List[Dict[str, Any]],
    output_path: str,
    data_all: List[Dict[str, Any]],
    short_model_name: str,
    long_model_name: str,
    process_num: int,
) -> List[int]:
    """Distribute SQL generation across a thread pool and return worker results."""
    contents: List[int] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                generate_sql,
                i * len(ids) // process_num,
                min((i + 1) * len(ids) // process_num, len(ids)),
                ids,
                dev,
                i,
                output_path,
                data_all,
                short_model_name,
                long_model_name,
            )
            for i in range(process_num)
        ]
        for future in concurrent.futures.as_completed(futures):
            contents.append(future.result())
    return contents


def main(options: argparse.Namespace) -> None:
    """Generate SQL for all samples in parallel, then merge chunk files into one output."""
    log(f"Starting: dev_path={options.dev_path}, output_path={options.output_path}")

    with open(options.dev_path) as f:
        dev = json.load(f)
    with open(options.data_path) as f:
        data_all = json.load(f)

    output_path = options.output_path
    if os.path.exists(output_path):
        log(f"Output {output_path} already exists, skipping")
        return

    generate_parallel(
        range(len(dev)),
        dev,
        output_path,
        data_all,
        options.short_model_name,
        options.long_model_name,
        options.process_num,
    )

    log(f"Merging {options.process_num} chunk files into {output_path}")
    with open(output_path, "w") as f_out:
        for i in range(options.process_num):
            chunk_path = get_output_name(output_path, i)
            with open(chunk_path, "r") as f_chunk:
                data = f_chunk.readlines()
            os.remove(chunk_path)
            for pre_sql in data:
                f_out.write(pre_sql.strip("\n") + "\n")

    log(f"Finished; saved SQL to {output_path}")


if __name__ == "__main__":
    opt = parse_option()
    main(opt)
