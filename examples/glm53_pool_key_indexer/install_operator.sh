#!/usr/bin/env bash
set -euo pipefail

package=${PKI_PACKAGE_DIR:-/mnt/share_space/d00954433/profiling/pki_c8_int8perf_v2_final/package}
install_root=/opt/pki_c8_int8perf_v2_final
vendor="$install_root/vendors/pool_key_indexer_transformer"
run="$package/cann-ops-transformer-pool_key_indexer_linux-aarch64.run"
wheel="$package/cann_ops_transformer_pool_key_indexer-1.0.0-py3-none-any.whl"

echo '== package MD5 =='
md5sum "$run" "$wheel"
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
