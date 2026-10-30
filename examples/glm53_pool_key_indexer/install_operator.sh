#!/usr/bin/env bash
set -euo pipefail

package=${PKI_PACKAGE_DIR:?Set PKI_PACKAGE_DIR to the downloaded pki_c8_int8perf_v3_ldgate directory}
install_root=${PKI_INSTALL_ROOT:-/opt/pki_c8_int8perf_v3_ldgate}
vendor="$install_root/vendors/pool_key_indexer_transformer"
run="$package/cann-ops-transformer-pool_key_indexer_linux-aarch64.run"
wheel="$package/cann_ops_transformer_pool_key_indexer-1.0.0-py3-none-any.whl"

echo '== verify v3 customer package SHA256 =='
printf '%s  %s\n' \
  c7075f2175ab888abed63ad31bb85e07c84a7b0bbfe0ba3013bbeaa0c2728550 "$run" \
  238c4320b19c60b252d9bdaf559d41cef8feb4e6c11083a634fe43a8bc9028d2 "$wheel" | sha256sum -c -
mkdir -p "$install_root"

echo '== install .run =='
bash "$run" --install-path="$install_root"

echo '== install .whl =='
python3 -m pip install "$wheel" --force-reinstall --no-deps

cat > /etc/profile.d/pki_mcy.sh <<EOF
source $vendor/bin/set_env.bash
export ASCEND_CUSTOM_OPP_PATH=$vendor
EOF
chmod 644 /etc/profile.d/pki_mcy.sh

config=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/config.ini
printf 'load_priority=pool_key_indexer_transformer,pki_drain_transformer,batch_invariant\n' > "$config"

echo '== verify files and Python binding =='
test -f "$vendor/op_api/lib/libcust_opapi.so"
test -f "$vendor/op_api/include/aclnnop/aclnn_pool_key_indexer.h"
test -f "$vendor/op_api/include/aclnnop/aclnn_pool_key_indexer_v2.h"
find "$vendor/op_impl/ai_core/tbe/kernel/ascend910_93/pool_key_indexer" -name '*.o' -printf '%f %s bytes\n' | sort
nm -D "$vendor/op_api/lib/libcust_opapi.so" | grep aclnnPoolKeyIndexer
python3 -c 'import cann_ops_transformer_pool_key_indexer as p; print("binding:", p.__file__, "callable:", callable(p.pool_key_indexer))'
echo "vendor=$vendor"
echo "load_priority=$(cat "$config")"
