#!/usr/bin/env bash
# 用法：bash run_p.sh <visible_devices> <http_port> <dp_size> \
# <dp_rank> <dp_address> <rpc_port> <tp_size>
# This A3 supernode uses HCCS; its device RoCE ports are down.
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

LOCAL_IP=${LOCAL_IP:-80.5.9.103}
NIC_NAME=${NIC_NAME:-enp194s0f0}
MODEL_PATH=${MODEL_PATH:-/mnt/share/w00936111/weights/GLM-5.3-Flash-W8A8-1010-V1}
PROFILE_ROOT=${PROFILE_ROOT:-/mnt/share/d00954433/pki_v3_103_110_20261029/staging/profile_p}

export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V2_MODEL_RUNNER=0
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export ASCEND_RT_VISIBLE_DEVICES="$1"
export VLLM_ASCEND_GLM5_KPOOL_INDEXER_MODE="${VLLM_ASCEND_GLM5_KPOOL_INDEXER_MODE:-builtin}"
if [[ -n "${PKI_VENDOR_PATH:-}" ]]; then
  export ASCEND_CUSTOM_OPP_PATH="$PKI_VENDOR_PATH"
  source "$PKI_VENDOR_PATH/bin/set_env.bash"
fi
DP_ARGS=()
if (( $3 > 1 )); then
  DP_ARGS=(--data-parallel-size "$3" --data-parallel-rank "$4" --data-parallel-address "$5" --data-parallel-rpc-port "$6")
fi

exec vllm serve "$MODEL_PATH" \
--host 0.0.0.0 \
--port "$2" \
"${DP_ARGS[@]}" \
--tensor-parallel-size "$7" \
--enable-expert-parallel \
--seed 1024 \
--served-model-name glm \
--safetensors-load-strategy prefetch \
--max-num-seqs 64 \
--max-model-len 262144 \
--max-num-batched-tokens 16384 \
--trust-remote-code \
--quantization ascend \
--limit-mm-per-prompt '{"image": 0, "video": 0}' \
--gpu-memory-utilization 0.92 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice \
--kv-cache-dtype int8 \
--prefix_cache_retention_interval None \
--enforce-eager \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
--additional_config '{"enable_flashcomm1": true,
   "enable_dsa_cp": true,
   "enable_prefill_mc2": true,
   "enable_fused_mc2": 1,
   "enable_shared_expert_dp": true,
   "multistream_overlap_shared_expert": false}' \
--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$PROFILE_ROOT\", \"torch_profiler_with_stack\": false}" \
--kv-transfer-config \
'{"kv_connector": "MooncakeConnectorV2", "kv_role": "kv_producer", "kv_port": "36680"}'
