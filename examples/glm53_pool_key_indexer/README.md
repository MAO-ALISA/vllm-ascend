# GLM-5.3 Flash PoolKeyIndexer 双机 PD 性能复现

这组脚本整理了 2026-10-28 至 29 日的测试配置。服务端使用两台各 16 卡的 A3 主机：P 为 `80.48.29.107`，D 为 `80.48.29.108`；压测端为 `80.5.17.120` 的 `aisbench_djc` 容器。地址、镜像、模型路径和挂载目录均可按环境修改。脚本不包含登录凭据。开始前确认设备空闲，并仅在本次测试容器中运行停止命令。

自定义 C8/C16 模式还需要安装 [模型接入分支](https://github.com/MAO-ALISA/vllm-ascend/tree/codex/pool-key-indexer-c8-integration) 的代码；`main` 当前没有 `VLLM_ASCEND_GLM5_KPOOL_INDEXER_MODE` 开关。本目录保存测试与部署脚本，不会仅凭安装算子包就让模型调用自定义算子。该分支与本目录合用尚未在 NPU 上重新端到端验证。

## 1. 创建容器和安装算子

两台服务主机分别执行。镜像是当时 `docker inspect` 得到的实际镜像标签。脚本使用 `--privileged`、host 网络、32 GiB 共享内存，并挂载模型和 Ascend 驱动。

```bash
export WORK_ROOT=/data/mcy_pki_c8_int8perf_v2_final
mkdir -p "$WORK_ROOT/scripts" "$WORK_ROOT/profile" "$WORK_ROOT/service_logs"
bash create_container.sh
cp ./*.sh ./*.py "$WORK_ROOT/scripts/"
docker exec glm53_flash_kpool_indexer_test_mcy bash "$WORK_ROOT/scripts/install_operator.sh"
```

`create_container.sh` 可用 `VLLM_IMAGE`、`CONTAINER_NAME`、`WORK_ROOT` 覆盖默认值。`install_operator.sh` 默认读取 `/mnt/share_space/d00954433/profiling/pki_c8_int8perf_v2_final/package`，也可通过 `PKI_PACKAGE_DIR` 指定包目录。它依次安装 `.run` 和 `.whl`，检查 Python binding、`aclnnPoolKeyIndexer` 符号，并在测试容器内设置自定义 OPP 优先级。算子包来源为 [pki_c8_int8perf_v2_final](https://gitcode.com/guijianwei/blueFIle/tree/main/poolkeyinderer/pki_c8_int8perf_v2_final)。

在两台容器中将模型接入分支安装为当前 `vllm_ascend`，确认 Python 实际加载的路径与版本；具体安装方式取决于镜像是否已做 editable 安装。示例：

```bash
git clone --branch codex/pool-key-indexer-c8-integration \
  https://github.com/MAO-ALISA/vllm-ascend.git /workspace/vllm-ascend-pki
python3 -m pip install -e /workspace/vllm-ascend-pki --no-deps
python3 -c 'import vllm_ascend; print(vllm_ascend.__file__)'
```

## 2. 启动 PD 分离服务

P 主机运行 `start-p`（2 个 DP rank，每 rank TP8）；D 主机运行 `start-d`（8 个 DP rank，每 rank TP2）和 `proxy`。两个主机的环境变量应使用相同 P/D 地址。每次切换 `builtin`、`c8`、`c16` 时，先在两台机器分别停止本次服务，再以同一模式启动，避免缓存和进程混杂。

```bash
export WORK_ROOT=/data/mcy_pki_c8_int8perf_v2_final
export PREFILL_HOST=80.48.29.107 DECODE_HOST=80.48.29.108
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
| P4K | 4096 / 1 | 0% | 16（每 DP 8） | 32 | 无 |
| P128K | 131072 / 1 | 90% | 4（每 DP 2） | 8 | 117964 / 1，2 条 |
| D4K | 4096 / 2048 | 100% | 80（每 DP 10） | 80 | 4096 / 1，2 条 |
| D128K | 131072 / 1024 | 100% | 80（每 DP 10） | 80 | 131072 / 1，2 条 |

上述命中率是请求构造目标，实际命中率应结合压测前后的 prefix cache metrics 核对。四个场景分别跑三种模式，共 12 组。`mode`、`scenario` 参数分别取 `builtin|c8|c16`、`p4k|p128k|d4k|d128k`。先以 `run_aisbench_case.sh` 检查常规 QPS、TTFT、ITL，再以 `profile_aisbench_case.py` 在单个 DP rank 开启短时间 profiling：

```bash
cd /mnt/share/d00954433/test_tools/aisbench_auto_tools_prefix
bash run_aisbench_case.sh c8 d128k warmup
bash run_aisbench_case.sh c8 d128k main
python3 profile_aisbench_case.py c8 d128k --prefill-host 80.48.29.107 --decode-host 80.48.29.108 --rank-limit 1 --profile-seconds 3
```

`profile_aisbench_case.py` 会自行执行 warmup 和 main，故单独采 profile 时只需最后一条命令。P 侧 trace 在 P 主机，D 侧 trace 在 D 主机。开始每个 case 前在对应主机执行 `touch "$WORK_ROOT/profile/.before_c8_d128k"`；压测后执行 `bash "$WORK_ROOT/scripts/export_profile_case.sh" c8 d128k <预期trace目录数>`，生成 `operator_summary.json`。导出脚本会检查 trace 数量，不满足时不会汇总。比较 `PoolKeyIndexer` 前先确认 trace 中确实存在该算子调用。

脚本中的默认目录和端口复刻当时测试环境；复制到其他环境时需同步修改容器挂载、数据集文件名和服务地址。所有模式都应记录代码提交、算子包校验值、服务日志、实际 prefix cache 命中率与 ais-bench 结果，便于公平比较。
