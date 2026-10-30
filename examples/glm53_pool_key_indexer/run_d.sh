#!/usr/bin/env bash

# 用法与 run_p.sh 相同。
# This A3 supernode uses HCCS; its device RoCE ports are down.
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

LOCAL_IP=${LOCAL_IP:-80.5.9.110}
NIC_NAME=${NIC_NAME:-enp48s3u1u1}
MODEL_PATH=${MODEL_PATH:-/mnt/share/w00936111/weights/GLM-5.3-Flash-W8A8-1010-V1}
PROFILE_ROOT=${PROFILE_ROOT:-/mnt/share/d00954433/pki_v3_103_110_20261029/staging/profile_d}

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

exec vllm serve "$MODEL_PATH" \
--host 0.0.0.0 \
--port "$2" \
--data-parallel-size "$3" \
--data-parallel-rank "$4" \
--data-parallel-address "$5" \
--data-parallel-rpc-port "$6" \
--tensor-parallel-size "$7" \
--enable-expert-parallel \
--seed 1024 \
--served-model-name glm \
--safetensors-load-strategy prefetch \
--max-num-seqs 32 \
--max-model-len 262144 \
--max-num-batched-tokens 192 \
--trust-remote-code \
--quantization ascend \
--limit-mm-per-prompt '{"image": 0, "video": 0}' \
--gpu-memory-utilization 0.92 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice \
--kv-cache-dtype int8 \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": false}' \
--additional_config '{"multistream_overlap_shared_expert": true, "enable_shared_expert_dp": true,"recompute_scheduler_enable": true, "enable_flashcomm1": true}' \
--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$PROFILE_ROOT\", \"torch_profiler_with_stack\": false}" \
--kv-transfer-config \
'{"kv_connector": "MooncakeConnectorV2", "kv_role": "kv_consumer", "kv_port": "36580"}'
