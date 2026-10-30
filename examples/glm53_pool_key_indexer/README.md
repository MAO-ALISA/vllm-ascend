# GLM-5.3 Flash PoolKeyIndexer 双机 PD 性能复现

这组脚本基于 vllm-ascend `b5e844b164ddefb97c0b44834336b6e2866db47c` 的 GLM-5.3 Flash 代码。默认 P 主机为 `80.5.9.103`、D 主机为 `80.5.9.110`，各有 16 张 A3 卡；地址、网卡、镜像、模型路径和挂载目录均可按环境修改。脚本不包含登录凭据。开始前确认设备空闲，并仅在本次测试容器中运行停止命令。

本补丁同时加入 `VLLM_ASCEND_GLM5_KPOOL_INDEXER_MODE` 模型开关。`builtin` 使用 CANN 内置算子，`c8` 使用自定义 INT8 算子，`c16` 使用自定义 BF16 算子。自定义路径显式使用 V2 索引布局；v3 新增的 `int8-only-ld-gate` 还要求运行时装箱损失达到 10%，A3 主形态按包文档预计不触发。模型代码、`.run` 和 `.whl` 必须同时安装在 P、D 两侧。

## 1. 创建容器和安装算子

两台服务主机分别执行。先设置 `VLLM_IMAGE` 为当前 b5e844 代码对应且与算子包 CANN 9.3.0 兼容的镜像。脚本使用 `--privileged`、host 网络、32 GiB 共享内存，并挂载模型和 Ascend 驱动。

```bash
export WORK_ROOT=/mnt/share/d00954433/pki_v3_103_110_20261029/staging
export VLLM_IMAGE='your-vllm-ascend-cann9.3-image'
mkdir -p "$WORK_ROOT/scripts" "$WORK_ROOT/profile_p" "$WORK_ROOT/profile_d" "$WORK_ROOT/service_logs"
bash create_container.sh
cp ./*.sh ./*.py "$WORK_ROOT/scripts/"
export PKI_PACKAGE_DIR=/mnt/share/d00954433/pki_v3_103_110_20261029/package
docker exec -e PKI_PACKAGE_DIR="$PKI_PACKAGE_DIR" glm53_flash_kpool_indexer_test_mcy \
  bash "$WORK_ROOT/scripts/install_operator.sh"
```

`create_container.sh` 可用 `VLLM_IMAGE`、`CONTAINER_NAME`、`WORK_ROOT` 覆盖默认值。`PKI_PACKAGE_DIR` 必须指向下载后的 v3 包目录；`install_operator.sh` 校验主交付 `.run` 和 `.whl` 的 SHA256，然后安装两者并检查 Python binding、aclnn 符号与 kernel 文件。默认安装根目录为 `/opt/pki_c8_int8perf_v3_ldgate`。算子包来源为 [pki_c8_int8perf_v3_ldgate](https://gitcode.com/guijianwei/blueFIle/tree/main/poolkeyinderer/pki_c8_int8perf_v3_ldgate)，此处使用客户主交付臂，不使用 `dbgverify` 臂。该包按 CANN 9.3.0 构建，部署前应确认运行环境兼容。

在两台容器中使用应用本补丁后的代码，确认 Python 实际加载的路径与版本；具体安装方式取决于镜像是否已做 editable 安装。示例：

```bash
git clone --branch scripts https://github.com/MAO-ALISA/vllm-ascend.git /workspace/vllm-ascend-pki
cd /workspace/vllm-ascend-pki
git apply /path/to/glm53_pki_v3_scripts.patch
python3 -m pip install -e /workspace/vllm-ascend-pki --no-deps
python3 -c 'import vllm_ascend; print(vllm_ascend.__file__)'
```

## 2. 启动 PD 分离服务

P 主机运行 `start-p`（2 个 DP rank，每 rank TP8）；D 主机运行 `start-d`（8 个 DP rank，每 rank TP2）和 `proxy`。两个主机的环境变量应使用相同 P/D 地址。每次切换 `builtin`、`c8`、`c16` 时，先在两台机器分别停止本次服务，再以同一模式启动，避免缓存和进程混杂。

```bash
export WORK_ROOT=/mnt/share/d00954433/pki_v3_103_110_20261029/staging
export PREFILL_HOST=80.5.9.103 DECODE_HOST=80.5.9.110
export PKI_MODE=c8  # builtin / c8 / c16

# P 主机
bash "$WORK_ROOT/scripts/control_pd.sh" start-p

# D 主机
bash "$WORK_ROOT/scripts/control_pd.sh" start-d
bash "$WORK_ROOT/scripts/control_pd.sh" proxy

# 两台主机分别查看服务日志和 /health
bash "$WORK_ROOT/scripts/control_pd.sh" status
```

`run_p.sh` 设置 `--max-num-batched-tokens 16384`；`run_d.sh` 设置 `192`，并使用 `FULL_DECODE_ONLY`。两者均启用 Mooncake PD KV 传输和 torch profiler。可用 `MODEL_PATH`、`NIC_NAME`、`LOCAL_IP`、`PROFILE_ROOT` 覆盖默认值。默认地址和网卡只适用于上述测试主机。

## 3. 安装 ais-bench 和准备测试容器

当时压测端使用 `ais_bench_benchmark==3.1.20261010`，源码提交 `8affdc1bccfa0437a912c757db4fe7a25b0329d2`，以 editable 方式安装。以下命令在 `aisbench_djc` 中执行。压测脚本还依赖已有的 `aisbench_auto_tools_prefix` 项目中的 `generate_dataset.py`、`save_file.py`、`cal_prefix_hit_rate.py` 及其数据集目录。

```bash
git clone https://github.com/AISBench/benchmark.git /mnt/share/d00954433/test_tools/benchmark
cd /mnt/share/d00954433/test_tools/benchmark
git checkout 8affdc1bccfa0437a912c757db4fe7a25b0329d2
python3 -m pip install -e .
ais_bench --help

tools=/mnt/share/d00954433/test_tools/aisbench_auto_tools_prefix
cp /path/to/this-directory/{mcy_aisbench_test_reuse.py,mcy_prefix_snapshot.py,run_aisbench_case.sh,profile_aisbench_case.py} "$tools/"
cp /path/to/this-directory/config.py.example "$tools/config.py"
```

按实际路径与地址修改 `config.py` 中的 `DATASET_PATH`、`WORK_PATH`、`MODEL_PATH`、`HOST_IP`、`HOST_PORT`、`POD_INFO`。`mcy_datasets` 下的 JSONL 文件要事先存在；文件名与长度、条数、重复率在 `run_aisbench_case.sh` 中列出。前缀数据集由相同 tokenizer 生成，warmup 和主测必须共享相同前缀。`POD_INFO` 和 `mcy_prefix_snapshot.py` 的 `PREFILL_HOST` 用于读取缓存命中指标。

## 4. 测试矩阵与 profiling

| 场景 | 输入 / 输出 token | 命中目标 | 总并发 | 请求数 | 前置请求 |
| --- | ---: | ---: | ---: | ---: | --- |
| P4K | 4096 / 1 | 0% | 200 | 200 | 无 |
| P128K | 131072 / 1 | 90% | 200 | 200 | 117964 / 1，2 条 |
| D4K | 4096 / 2048 | 100% | 80（每 DP 10） | 80 | 4096 / 1，8 条（每 DP 一条） |
| D128K | 131072 / 1024 | 100% | 80（每 DP 10） | 80 | 131072 / 1，8 条（每 DP 一条） |

上述命中率是请求构造目标，实际命中率应结合压测前后的 prefix cache metrics 核对。D 侧 8 条预热请求必须分别落到 8 个 DP rank，且主测数据应在每个 rank 复用对应预热前缀；仅有 8 条预热请求不能自动保证 100% 命中。四个场景分别跑三种模式，共 12 组。`mode`、`scenario` 参数分别取 `builtin|c8|c16`、`p4k|p128k|d4k|d128k`。先以 `run_aisbench_case.sh` 检查常规 QPS、TTFT、ITL，再以 `profile_aisbench_case.py` 在单个 DP rank 开启短时间 profiling：

```bash
cd /mnt/share/d00954433/test_tools/aisbench_auto_tools_prefix
bash run_aisbench_case.sh c8 d128k warmup
bash run_aisbench_case.sh c8 d128k main
python3 profile_aisbench_case.py c8 d128k --prefill-host 80.5.9.103 --decode-host 80.5.9.110 --rank-limit 1 --profile-seconds 3
```

`profile_aisbench_case.py` 会自行执行 warmup 和 main，故单独采 profile 时只需最后一条命令。P 侧 trace 在 `$WORK_ROOT/profile_p`，D 侧 trace 在 `$WORK_ROOT/profile_d`。开始每个 case 前在对应主机执行 `touch "$WORK_ROOT/profile_d/.before_c8_d128k"`；压测后执行 `bash "$WORK_ROOT/scripts/export_profile_case.sh" c8 d128k <预期trace目录数>`，生成 `operator_summary.json`。导出脚本会检查 trace 数量，不满足时不会汇总。比较 `PoolKeyIndexer` 前先确认 trace 中确实存在该算子调用。

脚本中的默认目录和端口复刻当时测试环境；复制到其他环境时需同步修改容器挂载、数据集文件名和服务地址。所有模式都应记录代码提交、算子包校验值、服务日志、实际 prefix cache 命中率与 ais-bench 结果，便于公平比较。

**C8 PD 正确性限制：** 当前实现将 INT8 pooled key 写入可传输的 KV 缓存，但对应的 per-pool FP16 scale 存在进程本地张量中，没有接入 Mooncake 的 KV 传输。跨机 P→D 后若 D 直接使用这些 key，scale 可能缺失。必须先补齐 scale 的传输并完成输出一致性验证，才能将 C8 的 PD 测试结果用于最终性能结论。C16 和 builtin 不使用这份 scale 缓存。本文脚本仅提供 C8 的接入与实验入口，不代表其 PD 正确性已经验证。
