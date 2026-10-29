#!/usr/bin/env python3
"""Aggregate PoolKeyIndexer device times from exported Ascend op_summary CSVs."""

import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", nargs="+")
    args = parser.parse_args()
    groups = defaultdict(list)
    overall = defaultdict(list)
    sources = []
    for trace in args.trace_dir:
        files = glob.glob(str(Path(trace) / "PROF_*" / "mindstudio_profiler_output" / "op_summary*.csv"))
        if len(files) != 1:
            raise RuntimeError(f"expected one op_summary CSV under {trace}, got {files}")
        sources.extend(files)
        rank = Path(trace).name.split("_20")[0]
        with open(files[0], encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                name = row["Op Name"]
                op_type = row["OP Type"]
                if not any(x in (name + " " + op_type).lower() for x in
                           ("poolkeyindexer", "pool_key_indexer", "keypool")):
                    continue
                key = (rank, name, op_type, row["Input Data Types"], row["Input Shapes"])
                duration = float(row["Task Duration(us)"])
                groups[key].append(duration)
                overall[(name, row["Input Data Types"])].append(duration)
    results = []
    for (rank, name, op_type, dtype, shape), values in sorted(groups.items()):
        results.append({"rank": rank, "name": name, "op_type": op_type,
                        "input_dtype": dtype, "input_shape": shape,
                        "count": len(values), "median_us": round(statistics.median(values), 3),
                        "mean_us": round(statistics.mean(values), 3),
                        "p90_us": round(percentile(values, 0.9), 3),
                        "sum_us": round(sum(values), 3)})
    totals = []
    for (name, dtype), values in sorted(overall.items()):
        totals.append({"name": name, "input_dtype": dtype, "count": len(values),
                       "median_us": round(statistics.median(values), 3),
                       "mean_us": round(statistics.mean(values), 3),
                       "p90_us": round(percentile(values, 0.9), 3),
                       "sum_us": round(sum(values), 3)})
    print(json.dumps({"sources": sources, "overall": totals, "groups": results},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
