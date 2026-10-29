#!/usr/bin/env bash
set -euo pipefail

mode=${1:?mode}
scenario=${2:?scenario}
phase=${3:?warmup or main}
root=${AISBENCH_TOOLS_ROOT:-/mnt/share/d00954433/test_tools/aisbench_auto_tools_prefix}
data=${AISBENCH_DATASET_ROOT:-$root/mcy_datasets}
model=${MODEL_BASENAME:-GLM-5.3-Flash-W8A8-1010-V1}
cd "$root"

case "$scenario" in
  p4k)
    test "$phase" = main || exit 0
    args=(--input_len 4096 --output_len 1 --data_num 32 --concurrency 16
      --dataset "$data/GSM8K-in4096-num32-$model.jsonl" --dp 2 --seed 5341)
    ;;
  p128k)
    if [[ "$phase" == warmup ]]; then
      args=(--input_len 117964 --output_len 1 --data_num 2 --concurrency 2
        --dataset "$data/prefix-GSM8K-in117964-num2-$model.jsonl" --dp 2 --seed 5342)
    else
      args=(--input_len 131072 --output_len 1 --data_num 8 --concurrency 4
        --dataset "$data/GSM8K-in131072-num8-$model-repeatRate0.9.jsonl" --dp 2 --seed 5342)
    fi
    ;;
  d4k)
    if [[ "$phase" == warmup ]]; then
      args=(--input_len 4096 --output_len 1 --data_num 2 --concurrency 2
        --dataset "$data/prefix-GSM8K-in4096-num2-$model.jsonl" --dp 2 --seed 5343)
    else
      args=(--input_len 4096 --output_len 2048 --data_num 80 --concurrency 80
        --dataset "$data/GSM8K-in4096-num80-$model-repeatRate1.0.jsonl" --dp 2 --seed 5343)
    fi
    ;;
  d128k)
    if [[ "$phase" == warmup ]]; then
      args=(--input_len 131072 --output_len 1 --data_num 2 --concurrency 2
        --dataset "$data/prefix-GSM8K-in131072-num2-$model.jsonl" --dp 2 --seed 5344)
    else
      args=(--input_len 131072 --output_len 1024 --data_num 80 --concurrency 80
        --dataset "$data/GSM8K-in131072-num80-$model-repeatRate1.0.jsonl" --dp 2 --seed 5344)
    fi
    ;;
  *) echo "unknown scenario: $scenario" >&2; exit 2 ;;
esac

if [[ "$phase" == main ]]; then
  python3 mcy_prefix_snapshot.py > "mcy_profile_${mode}_${scenario}_prefix_before.json" 2>&1 || true
fi
python3 mcy_aisbench_test_reuse.py "${args[@]}" --request_rate 0 --npu_num 16 \
  > "mcy_profile_${mode}_${scenario}_${phase}.log" 2>&1
if [[ "$phase" == main ]]; then
  python3 mcy_prefix_snapshot.py > "mcy_profile_${mode}_${scenario}_prefix_after.json" 2>&1 || true
fi
echo "$mode $scenario $phase complete"
tail -n 1 aisbench_result.csv
