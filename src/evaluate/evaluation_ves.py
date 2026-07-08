import argparse
import json
import math
import multiprocessing as mp
import os
import pdb
import sqlite3
import sys
import time

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout


def result_callback(result):
    exec_result.append(result)


def clean_abnormal(input):
    input = np.asarray(input)
    processed_list = []
    mean = np.mean(input, axis=0)
    std = np.std(input, axis=0)
    for x in input:
        if x < mean + 3 * std and x > mean - 3 * std:
            processed_list.append(x)
    return processed_list


def execute_sql(sql, db_path):
    # Connect to the database
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    start_time = time.time()
    cursor.execute(sql)
    exec_time = time.time() - start_time
    return exec_time


def iterated_execute_sql(predicted_sql, ground_truth, db_path, iterate_num):
    conn = sqlite3.connect(db_path)
    diff_list = []
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    time_ratio = 0
    if set(predicted_res) == set(ground_truth_res):
        for i in range(iterate_num):
            predicted_time = execute_sql(predicted_sql, db_path)
            ground_truth_time = execute_sql(ground_truth, db_path)
            diff_list.append(ground_truth_time / predicted_time)
        processed_diff_list = clean_abnormal(diff_list)
        time_ratio = sum(processed_diff_list) / len(processed_diff_list)
    return time_ratio


def execute_model(
    predicted_sql, ground_truth, db_place, idx, iterate_num, meta_time_out
):
    try:
        # you can personalize the total timeout number
        # larger timeout leads to more stable ves
        # while it needs more your patience....
        time_ratio = func_timeout(
            meta_time_out * iterate_num,
            iterated_execute_sql,
            args=(predicted_sql, ground_truth, db_place, iterate_num),
        )
        # print([idx, math.sqrt(time_ratio)])
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f"timeout",)]
        time_ratio = 0
    except Exception as e:
        result = [(f"error",)]  # possibly len(query) > 512 or not executable
        time_ratio = 0
    result = {"sql_idx": idx, "time_ratio": time_ratio}
    return result


def package_sqls(sql_path, db_root_path, mode="gpt", data_mode="dev"):
    clean_sqls = []
    db_path_list = []
    if mode == "gpt":
        sql_data = json.load(open(sql_path + "predict_" + data_mode + ".json", "r"))
        for idx, sql_str in sql_data.items():
            if type(sql_str) == str:
                sql, db_name = sql_str.split("\t----- bird -----\t")
            else:
                sql, db_name = " ", "financial"
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + "/" + db_name + ".sqlite")

    elif mode == "gt":
        sqls = open(sql_path + data_mode + "_gold.sql")
        sql_txt = sqls.readlines()
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split("\t")
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + "/" + db_name + ".sqlite")

    return clean_sqls, db_path_list


def run_sqls_parallel(sqls, db_places, num_cpus=1, iterate_num=100, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(
            execute_model,
            args=(
                predicted_sql,
                ground_truth,
                db_places[i],
                i,
                iterate_num,
                meta_time_out,
            ),
            callback=result_callback,
        )
    pool.close()
    pool.join()


def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x["sql_idx"])


def _safe_div(numerator: float, denominator: int, label: str = "") -> float:
    """Return numerator/denominator, or 0.0 when denominator is zero.

    A zero denominator means the current dataset slice contains no queries
    for that difficulty bucket.  We warn the user rather than crashing so
    that small subsets used for debugging still produce a valid report.
    """
    if denominator == 0:
        if label:
            print(
                f'[WARNING] No "{label}" samples in this subset — '
                f"reporting 0.00 VES for that difficulty level."
            )
        return 0.0
    return numerator / denominator


def compute_ves(exec_results, label: str = "") -> float:
    num_queries = len(exec_results)
    total_ratio = 0
    count = 0

    for i, result in enumerate(exec_results):
        if result["time_ratio"] != 0:
            count += 1
        total_ratio += math.sqrt(result["time_ratio"]) * 100
    return _safe_div(total_ratio, num_queries, label)


def load_json(dir):
    with open(dir, "r") as j:
        contents = json.loads(j.read())
    return contents


def infer_difficulty(content):
    difficulty = content.get("difficulty")
    if difficulty is not None:
        return difficulty

    sql = content.get("query") or content.get("SQL") or ""
    sql = sql.lower()
    last_sql = sql
    while sql != last_sql:
        sql = sql.replace("  ", " ")
        last_sql = sql
    sql = sql.replace("(*)", "").replace("( * )", "")
    sql_split = sql.split()

    tables = []
    for i in range(len(sql_split)):
        if sql_split[i] == "from" or sql_split[i] == "join":
            tables.append(sql_split[i + 1])

    where_count = 0
    unit_count = 0
    agg_count = 0
    cond_count = 0
    sql_count = 0
    order_count = 0

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
    UNIT_OPS = ("-", "+", "*", "/")
    AGG_OPS = ("max", "min", "count", "sum", "avg")
    COND_OPS = ("and", "or")
    SQL_OPS = ("intersect", "union", "except")
    ORDER_OPS = ("desc", "asc")

    for where_ops in WHERE_OPS:
        if where_ops in sql:
            where_count += sql.count(where_ops)
    for unit_ops in UNIT_OPS:
        if unit_ops in sql:
            unit_count += sql.count(unit_ops)
    for agg_ops in AGG_OPS:
        if agg_ops in sql:
            agg_count += sql.count(agg_ops)
    for cond_ops in COND_OPS:
        if cond_ops in sql:
            cond_count += sql.count(cond_ops)
    for sql_ops in SQL_OPS:
        if sql_ops in sql:
            sql_count += sql.count(sql_ops)
    for order_ops in ORDER_OPS:
        if order_ops in sql:
            order_count += sql.count(order_ops)

    all_count = (
        where_count + unit_count + agg_count + cond_count + sql_count + order_count
    )

    tables = set(tables)

    if len(tables) == 1 and all_count <= 6:
        return "simple"
    if len(tables) > 1 and all_count <= 6:
        return "challenging"
    if len(tables) == 1 and all_count > 6:
        return "moderate"
    return "challenging"


def compute_ves_by_diff(exec_results, diff_json_path):
    num_queries = len(exec_results)
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []
    # zip stops at the shorter list, so a subset exec_results never
    # causes an IndexError when contents comes from the full dev.json.
    for result, content in zip(exec_results, contents):
        difficulty = infer_difficulty(content)
        if difficulty == "simple":
            simple_results.append(result)
        if difficulty == "moderate":
            moderate_results.append(result)
        if difficulty == "challenging":
            challenging_results.append(result)
    simple_ves = compute_ves(simple_results, label="simple")
    moderate_ves = compute_ves(moderate_results, label="moderate")
    challenging_ves = compute_ves(challenging_results, label="challenging")
    all_ves = compute_ves(exec_results)
    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        num_queries,
    ]
    return simple_ves, moderate_ves, challenging_ves, all_ves, count_lists


def print_data(score_lists, count_lists):
    levels = ["simple", "moderate", "challenging", "total"]
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format("count", *count_lists))

    print(
        "=========================================    VES   ========================================"
    )
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format("ves", *score_lists))


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--predicted_sql_path", type=str, required=True, default=""
    )
    args_parser.add_argument("--ground_truth_path", type=str, required=True, default="")
    args_parser.add_argument("--data_mode", type=str, required=True, default="dev")
    args_parser.add_argument("--db_root_path", type=str, required=True, default="")
    args_parser.add_argument("--num_cpus", type=int, default=1)
    args_parser.add_argument("--meta_time_out", type=float, default=30.0)
    args_parser.add_argument("--mode_gt", type=str, default="gt")
    args_parser.add_argument("--mode_predict", type=str, default="gpt")
    args_parser.add_argument("--diff_json_path", type=str, default="")
    args = args_parser.parse_args()
    exec_result = []

    pred_queries, db_paths = package_sqls(
        args.predicted_sql_path,
        args.db_root_path,
        mode=args.mode_predict,
        data_mode=args.data_mode,
    )
    # generate gt sqls:
    gt_queries, db_paths_gt = package_sqls(
        args.ground_truth_path, args.db_root_path, mode="gt", data_mode=args.data_mode
    )

    query_pairs = list(zip(pred_queries, gt_queries))
    run_sqls_parallel(
        query_pairs,
        db_places=db_paths,
        num_cpus=args.num_cpus,
        meta_time_out=args.meta_time_out,
    )
    exec_result = sort_results(exec_result)
    print("start calculate")
    simple_ves, moderate_ves, challenging_ves, ves, count_lists = compute_ves_by_diff(
        exec_result, args.diff_json_path
    )
    score_lists = [simple_ves, moderate_ves, challenging_ves, ves]
    print_data(score_lists, count_lists)
    print(
        "==========================================================================================="
    )
    print("Finished evaluation")
