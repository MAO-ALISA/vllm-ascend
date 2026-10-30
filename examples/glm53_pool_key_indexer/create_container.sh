#!/usr/bin/env bash
set -euo pipefail

# Run on each service host after checking that its NPUs are free.
container=${CONTAINER_NAME:-glm53_flash_kpool_indexer_test_mcy}
root=${WORK_ROOT:-/mnt/share/d00954433/pki_v3_103_110_20261029/staging}
image=${VLLM_IMAGE:?Set VLLM_IMAGE to the vllm-ascend b5e844 image with a CANN 9.3 compatible runtime}
mkdir -p "$root/scripts" "$root/profile_p" "$root/profile_d" "$root/service_logs"

docker run -d --name "$container" --privileged --network host \
  --ipc=shareable --shm-size=32g --security-opt label=disable \
  -v /mnt:/mnt:ro -v "$root:$root" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  "$image" sleep infinity
