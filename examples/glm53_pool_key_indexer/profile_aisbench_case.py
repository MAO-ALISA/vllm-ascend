#!/usr/bin/env python3
"""Run one existing ais_bench case with a bounded P or D profiler window.

Run inside aisbench_djc from aisbench_auto_tools_prefix after the PD service is up.
The benchmark's existing datasets and request counts remain unchanged.
"""

import argparse
import concurrent.futures
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path


def profile_action(url, action):
    request = urllib.request.Request(f"{url}/{action}", data=b"", method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=240) as response:
        return {"url": url, "action": action, "status": response.status,
                "body": response.read(1024).decode("utf-8", "replace")}


def profile_all(urls, action):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as pool:
        rows = list(pool.map(lambda url: profile_action(url, action), urls))
    if any(row["status"] != 200 for row in rows):
        raise RuntimeError(f"{action} failed: {rows}")
    return rows


def wait_running(url, process, timeout=90):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("benchmark finished before the profiler window")
        try:
            with opener.open(f"{url}/metrics", timeout=3) as response:
                body = response.read().decode("utf-8", "replace")
            match = re.search(r"^vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.]+)",
                              body, flags=re.MULTILINE)
            if match and float(match.group(1)) > 0:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"no running requests observed on {url} within {timeout}s")


def wait_posted(log_path, process, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("benchmark finished before requests were posted")
        if log_path.exists() and b"POST=" in log_path.read_bytes()[-16384:]:
            return
        time.sleep(0.2)
    raise RuntimeError(f"no posted requests observed in {log_path} within {timeout}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("builtin", "c8", "c16"))
    parser.add_argument("scenario", choices=("p4k", "p128k", "d4k", "d128k"))
    parser.add_argument("--profile-seconds", type=float, default=3)
    parser.add_argument("--prefill-delay-seconds", type=float, default=0)
    parser.add_argument("--rank-limit", type=int, default=1,
                        help="number of DP rank endpoints to profile, starting at DP0")
    parser.add_argument("--prefill-host", default="80.48.29.107")
    parser.add_argument("--decode-host", default="80.48.29.108")
    args = parser.parse_args()

    is_prefill = args.scenario.startswith("p")
    all_ports = (9181, 9182) if is_prefill else tuple(range(9900, 9908))
    if not 1 <= args.rank_limit <= len(all_ports):
        parser.error("rank-limit is outside available DP ranks")
    ports = all_ports[:args.rank_limit]
    host = args.prefill_host if is_prefill else args.decode_host
    urls = [f"http://{host}:{port}" for port in ports]
    output = Path("mcy_profile_results") / args.mode / args.scenario
    output.mkdir(parents=True, exist_ok=True)
    record = {"mode": args.mode, "scenario": args.scenario, "rank_urls": urls,
              "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "profile_seconds": args.profile_seconds,
              "rank_limit": args.rank_limit}
    started = False
    process = None
    try:
        with (output / "warmup.log").open("w", encoding="utf-8") as log:
            subprocess.run(["bash", "run_aisbench_case.sh", args.mode,
                            args.scenario, "warmup"], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        if is_prefill and args.prefill_delay_seconds == 0:
            started = True
            record["profile_start"] = profile_all(urls, "start_profile")
        with (output / "benchmark.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                ["bash", "run_aisbench_case.sh", args.mode,
                 args.scenario, "main"],
                stdout=log, stderr=subprocess.STDOUT)
            if is_prefill:
                wait_posted(Path(f"mcy_profile_{args.mode}_{args.scenario}_main.log"),
                            process)
                if args.prefill_delay_seconds:
                    time.sleep(args.prefill_delay_seconds)
                    started = True
                    record["profile_start"] = profile_all(urls, "start_profile")
                time.sleep(args.profile_seconds)
                record["profile_stop"] = profile_all(urls, "stop_profile")
                started = False
                process.wait()
            else:
                wait_running(urls[0], process)
                started = True
                record["profile_start"] = profile_all(urls, "start_profile")
                time.sleep(args.profile_seconds)
                record["profile_stop"] = profile_all(urls, "stop_profile")
                started = False
                process.wait()
        record["benchmark_exit_code"] = process.returncode
        if process.returncode:
            raise RuntimeError(f"ais_bench exited {process.returncode}")
    finally:
        if started:
            record["profile_stop"] = profile_all(urls, "stop_profile")
        record["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (output / "control.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
