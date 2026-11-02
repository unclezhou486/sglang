#!/usr/bin/env python3
"""Summarize topk=1 vs topk=2 serving benchmark JSONL files."""

import argparse
import glob
import json
import os
import statistics
import sys

FIELDS = (
    "accept_length",
    "output_throughput",
    "median_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "mean_ttft_ms",
    "completed",
)


def load_rows(out_dir: str, label: str):
    paths = sorted(glob.glob(os.path.join(out_dir, f"{label}_r*.jsonl")))
    if not paths:
        raise FileNotFoundError(
            f"no result files matching {label}_r*.jsonl under {out_dir}"
        )
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return paths, rows


def median(rows, field):
    values = [row.get(field) for row in rows if row.get(field) is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def print_summary(label, paths, rows):
    print(f"\n[{label}] {len(rows)} run(s), {len(paths)} file(s)")
    for field in FIELDS:
        print(f"  {field:<24} {fmt(median(rows, field))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--topk1-label", default="topk1")
    parser.add_argument("--topk2-label", default="topk2")
    args = parser.parse_args()

    try:
        top1_paths, top1_rows = load_rows(args.out_dir, args.topk1_label)
        top2_paths, top2_rows = load_rows(args.out_dir, args.topk2_label)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    print_summary(args.topk1_label, top1_paths, top1_rows)
    print_summary(args.topk2_label, top2_paths, top2_rows)

    top1_accept = median(top1_rows, "accept_length")
    top2_accept = median(top2_rows, "accept_length")
    top1_tput = median(top1_rows, "output_throughput")
    top2_tput = median(top2_rows, "output_throughput")
    top1_tpot = median(top1_rows, "median_tpot_ms")
    top2_tpot = median(top2_rows, "median_tpot_ms")
    top1_itl = median(top1_rows, "median_itl_ms")
    top2_itl = median(top2_rows, "median_itl_ms")

    print("\n[topk2 / topk1]")
    print(f"  accept_length           {fmt(ratio(top2_accept, top1_accept))}")
    print(f"  output_throughput       {fmt(ratio(top2_tput, top1_tput))}")
    print(f"  median_tpot_ms          {fmt(ratio(top2_tpot, top1_tpot))}")
    print(f"  median_itl_ms           {fmt(ratio(top2_itl, top1_itl))}")

    print("\nInterpretation:")
    print("  accept_length > 1 and output_throughput > 1: tree is winning")
    print("  accept_length > 1 but output_throughput <= 1: tree overhead too high")
    print("  accept_length <= 1: tree branch is not covering a better path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
