#!/usr/bin/env bash
set -euo pipefail

# Run on each service host after checking that its NPUs are free.
container=${CONTAINER_NAME:-glm53_flash_kpool_indexer_test_mcy}
root=${WORK_ROOT:-/data/mcy_pki_c8_int8perf_v2_final}
image=${VLLM_IMAGE:-harbor.computing.lab.rnd.huawei.com/hisecr/vllm-ascend:nightly-ci-releases-v0.30.0-a3-openeuler-cann9.2.0_20261022115200-20261024-r35}
mkdir -p "$root/scripts" "$root/profile" "$root/service_logs"

docker run -d --name "$container" --privileged --network host \
  --ipc=shareable --shm-size=32g --security-opt label=disable \
  -v /mnt:/mnt:ro -v "$root:$root" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  "$image" sleep infinity
