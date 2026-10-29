#!/usr/bin/env bash
set -euo pipefail

action=${1:?start-p, start-d, proxy, status, stop}
container=${CONTAINER_NAME:-glm53_flash_kpool_indexer_test_mcy}
root=${WORK_ROOT:-/data/mcy_pki_c8_int8perf_v2_final}
prefill_host=${PREFILL_HOST:-80.48.29.107}
decode_host=${DECODE_HOST:-80.48.29.108}
mode=${PKI_MODE:-c8}
case "$mode" in
  c8) vendor=/opt/pki_c8_int8perf_v2_final/vendors/pool_key_indexer_transformer ;;
  c16) vendor=/opt/pki_c8_int8perf_v2_final/vendors/pool_key_indexer_transformer ;;
  builtin) vendor= ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
logs=$root/service_logs
mkdir -p "$logs"

case "$action" in
  start-p)
    for rank in 0 1; do
      start=$((rank * 8))
      end=$((start + 7))
      devices=$(seq -s, "$start" "$end")
      port=$((9181 + rank))
      docker exec -d -e "VLLM_ASCEND_GLM5_KPOOL_INDEXER_MODE=$mode" -e "PKI_VENDOR_PATH=$vendor" -e "LOCAL_IP=$prefill_host" \
        "$container" bash -lc \
        "export VLLM_ASCEND_GLM5_KPOOL_INDEXER_MODE=$mode PKI_VENDOR_PATH='$vendor' LOCAL_IP='$prefill_host'; if [[ '$mode' == builtin ]]; then unset ASCEND_CUSTOM_OPP_PATH; fi; bash $root/scripts/run_p.sh '$devices' '$port' 2 '$rank' '$prefill_host' 12325 8 > '$logs/p$rank.log' 2>&1"
      echo "P rank=$rank devices=$devices port=$port log=$logs/p$rank.log"
    done
    ;;
  start-d)
    for rank in $(seq 0 7); do
      first=$((rank * 2))
      second=$((first + 1))
      port=$((9900 + rank))
      docker exec -d -e "VLLM_ASCEND_GLM5_KPOOL_INDEXER_MODE=$mode" -e "PKI_VENDOR_PATH=$vendor" -e "LOCAL_IP=$decode_host" \
        "$container" bash -lc \
        "export VLLM_ASCEND_GLM5_KPOOL_INDEXER_MODE=$mode PKI_VENDOR_PATH='$vendor' LOCAL_IP='$decode_host'; if [[ '$mode' == builtin ]]; then unset ASCEND_CUSTOM_OPP_PATH; fi; bash $root/scripts/run_d.sh '$first,$second' '$port' 8 '$rank' '$decode_host' 12326 2 > '$logs/d$rank.log' 2>&1"
      echo "D rank=$rank devices=$first,$second port=$port log=$logs/d$rank.log"
    done
    ;;
  proxy)
    docker exec -d "$container" bash -lc \
      "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; python3 $root/scripts/load_balance_proxy_server_example.py --host 0.0.0.0 --port 8090 --prefiller-hosts '$prefill_host' '$prefill_host' --prefiller-ports 9181 9182 --decoder-hosts '$decode_host' '$decode_host' '$decode_host' '$decode_host' '$decode_host' '$decode_host' '$decode_host' '$decode_host' --decoder-ports 9900 9901 9902 9903 9904 9905 9906 9907 > '$logs/proxy8090.log' 2>&1"
    echo "proxy port=8090 log=$logs/proxy8090.log"
    ;;
  status)
    for port in 9181 9182 9900 9901 9902 9903 9904 9905 9906 9907 8090; do
      code=$(curl --noproxy '*' -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$port/health" || true)
      echo "$port $code"
    done
    ;;
  stop)
    docker exec "$container" bash -lc "pkill -TERM -f '[v]llm serve' || true; pkill -TERM -f '[l]oad_balance_proxy_server_example.py' || true"
    ;;
  *) echo "unknown action $action" >&2; exit 2 ;;
esac
