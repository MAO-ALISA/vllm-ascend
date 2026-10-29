#!/usr/bin/env bash
set -euo pipefail

mode=${1:?mode}
scenario=${2:?scenario}
expected=${3:?expected TP trace count}
work_root=${WORK_ROOT:-/data/mcy_pki_c8_int8perf_v2_final}
container=${CONTAINER_NAME:-glm53_flash_kpool_indexer_test_mcy}
root=$work_root/profile
marker="$root/.before_${mode}_${scenario}"
out="$root/exports/${mode}/${scenario}"
test -f "$marker"
mkdir -p "$out"
mapfile -t traces < <(find "$root" -mindepth 1 -maxdepth 1 -type d -name '*_ascend_pt' -newer "$marker" | sort)
test "${#traces[@]}" -eq "$expected" || { echo "Expected $expected rank traces, got ${#traces[@]}" >&2; exit 1; }
printf '%s\n' "${traces[@]}" > "$out/trace_dirs.txt"

for trace in "${traces[@]}"; do
  docker exec "$container" bash -lc "msprof --parse=on --output='$trace' > '$trace/parse.log' 2>&1 && msprof --export=on --output='$trace' > '$trace/export.log' 2>&1" &
  while (( $(jobs -pr | wc -l) >= 4 )); do wait -n; done
done
wait

python3 "$work_root/scripts/summarize_profile_csv.py" "${traces[@]}" > "$out/operator_summary.json"
echo "$out/operator_summary.json"
