# 路线 A「省显存档」（llama.cpp + KVMem）· RTX 3060 12GB 实测

> 实测日期 **2026-10-03** ｜ 机器：**RTX 3060 12GB（sm_86 / 28 SM / ~360 GB/s）**，
> Ubuntu 26.04，驱动 595.91.07（驱动侧 CUDA 13.2），i5-12400F，30 GB 内存。
> 引擎：`kvmem/kvmem-llama.cpp` 的 `kvmem-bonsai-llama.cpp` 分支（Bonsai/prism.3 线），
> **本机自编 CUDA 13.2.86 / `-DCMAKE_CUDA_ARCHITECTURES=86-real`**（官方 Linux 预编译不含三元，见 §2）。

---

## 0. 一句话

在 12GB 的 30 系卡上，这条线拿到 **128K 逻辑上下文**（KVMem 把历史 KV 放主机内存、GPU 只驻留窗口），
**decode 34~41 t/s、prefill 206~233 t/s、峰值显存 8.0~8.8 GiB / 12 GB**，
并把一条藏在 **108,699 token** 提示里的信息准确召回。

---

## 1. 实测速览

| 项 | 实测 | 备注 |
|---|---|---|
| 逻辑上下文 | **131072（128K）** | KVMem：GPU 只驻留 24K 检索窗口 + 10K 生成预留 |
| decode（无投机） | **34.1 t/s**（中文 505 tok）/ 33.9（英文 700 tok） | 600 词长输出，贪心 |
| decode（MTP1） | **39.3 / 41.2 t/s**（+15~22%） | 社区 r3 MTP 头 → Q4_0 合并，草稿 KV f16 |
| decode（MTP2） | 34.6 / 38.6 t/s | **本卡上比 MTP1 慢**，不推荐 |
| prefill | 232.7 t/s @6.8k ｜ 208.9 @36k ｜ **205.7 @109k** | 长提示不明显衰减 |
| 长上下文召回 | **109k 提示 → 正确**（代码藏在 40k 与 56k 段落之间） | 单样本验证，不是矩阵 |
| 多轮前缀复用 | ✅ 命中（`prompt_cache_hit_tokens` 30+） | KVMem 核心机制 |
| 思考模式 | ✅ 正常（`reasoning_budget` 4096，答案正确） | 请求级可关 |
| 峰值显存 | **8.0 GiB**（无 MTP）/ **8.6~8.8 GiB**（MTP1） | `nvidia-smi` 实读，含桌面 1.2 GiB |
| 模型加载 | ~1.3 s（页缓存热）/ 6.95 GiB 权重 | PQT1_0 5.95 GB |
| 编译 | 525/525，**0 错误**，6 核 **7.7 分钟**（ninja 计时，只编 sm_86 一个架构） | 见 §3 |

基准脚本：**`scripts/bench_kvmem.py`**（本仓库自带，`python3 scripts/bench_kvmem.py` 直接出上表数字）。

### 补充实测：256K 档（`long` profile，2026-10-08 验证）

长会话会撞"提示 token + 预留 > `-c`"的硬墙（见 §7 坑 8）。把 `-c` 顶到**模型原生上限 262144** 后可解，
代价只有**主机内存**（历史 KV 不在显存里）：

| 项 | 实测（RTX 3060 12GB） |
|---|---|
| 引擎自报 | `n_ctx=262144`（work 档 131072 的 2 倍） |
| 提示上限 | `262144 − 10240 = 251904` token（原 120832） |
| **12.1 万 token 真请求** | ✅ 通过：预填充 **571 s / 120851 token（211.6 t/s）**，无 OOM |
| 重复同一请求（缓存热） | ✅ **2.9 s**，`cache = 120753 / 120851`，只补算 98 token |
| 显存 | 8.06 GiB（空载）→ 8.58 GiB（跑完 12 万 token 后）—— 与 128K 档基本相同 |
| 主机内存 | 30 GB 机器：9 GiB（空载）→ 15 GiB（装载 12 万 token KV 后） |

启动：`./scripts/start-kvmem.sh long`。注意超长上下文的召回质量是官方标注的实验特性，能装下 ≠ 都记得住。

---

## 2. 为什么路线 A 在本机也必须"自编"一次

这一点容易误判，先说清楚：

- 官方 **rc3 Linux 预编译包**（`kvmem-v0.16.0-rc3-linux-x86_64-cuda13.2.86.tar.gz`）**确实编了 sm_86**，
  但那套是**通用 Qwen3.8 线**（IQ3/IQ4），**不含 Bonsai 三元格式**
  （`general.file_type = 143` = `PTQ1_0`，外加 `prism.hadamard.*` 激活变换）。
- 带三元支持的 **prism.3 分支**，官方**只发了 Windows 预编译**
  （`...-windows-x86_64-cuda12.9-rtx20-30-40-50.zip`）。
- ⇒ Linux + 30 系 + 三元 = **只能自编**（本页做的就是这个）。好消息：编译产物跑起来后
  **不需要系统装 CUDA Toolkit**（动态库都在工作目录里，见 §6 的 `LD_LIBRARY_PATH`）。

---

## 3. 免 root 工具链（本机没有 sudo 密码，全部装在用户目录）

CUDA 13.2 Update 2（`13.2.86`，与官方 rc3/prism.3 构建基线一致）用 **NVIDIA apt 仓库的 .deb 直接解包**，
不装系统包、不改 `/usr/local`：

```bash
# 关键：libnvvm 那个包不能漏，否则 nvcc 报 "nvvm/bin/cicc: not found"
# 清单：nvcc / crt / cudart(-dev) / cccl / driver-dev / culibos(-dev) / nvrtc(-dev)
#       libcublas(-dev) / libnvvm / cuda-tileiras / libnvjitlink(-dev) / libnvfatbin
BASE=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
for f in cuda-nvcc-13-2_13.2.86-1_amd64.deb libnvvm-13-2_13.2.86-1_amd64.deb ...; do
  curl -fsSL --noproxy '*' -o "debs/$f" "$BASE/$f" && dpkg -x "debs/$f" cuda-stage/
done
# 结果：cuda-stage/usr/local/cuda-13.2/ 就是一棵完整、可用的工具链
```

验证（本机通过）：

```bash
nvcc --version           # Cuda compilation tools, release 13.2, V13.2.86
nvcc -arch=sm_86 -o t t.cu && ./t    # → ok 20.085537（真在 3060 上跑）
```

- cmake 3.31.6（cmake.org tarball）+ ninja 1.12.1（GitHub via ghfast 镜像），同样只解压到 `tools/`。
- 宿主编译器用系统 **GCC 15.2**，nvcc 13.2 直接接受，无需 `-allow-unsupported-compiler`。

**完整一键脚本：`scripts/build-kvmem-sm86.sh`。**

---

## 4. 编译

```bash
git clone --branch kvmem-bonsai-llama.cpp --recurse-submodules --shallow-submodules \
  https://github.com/kvmem/kvmem-llama.cpp.git
cd kvmem-llama.cpp && ./scripts/apply-patches.sh      # 给 Prism 源打 KVMem 适配补丁（30 个文件）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86-real \                # ★ 30 系就这一处
  -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DKVMEM_BUILD_LLAMA=ON -DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT=$PWD
ninja -C build -j6 llama-kvmem-server llama-kvmem-cli
```

- `-j6` 不是保守，是**内存纪律**：nvcc 单作业峰值 1.5~3 GB，30 GB 内存机器上 `-j12` 会触顶
  （官方 Windows 侧就出过 `-j24` 打爆内存导致内核崩溃的事故，见 `汇总-报告…/死机诊断与安全续跑.md`）。
- `cmake` 配置输出里出现 `Found CUDAToolkit: ... 13.2.86` 与
  `CMAKE_CUDA_ARCHITECTURES=86-real` 即正确。

---

## 5. 启动（3060 12GB 档位）

`scripts/start-kvmem.sh`（本仓库）：

```bash
./scripts/start-kvmem.sh work            # 128K 上下文，GPU 窗口 24K+10K，KV q8_0，思考开
KVMEM_MTP=1 ./scripts/start-kvmem.sh work   # 打开 MTP1（需先合并 MTP 头，见下）

# Web UI 可选（源码树的 ui/ 是 Svelte 源，需 Node 构建；没有则只提供 API）
curl -s http://127.0.0.1:18202/v1/models
```

关键参数（来自官方 `start-bonsai.ps1`，逐项对应）：

| 参数 | 值 | 说明 |
|---|---|---|
| `-c` | 131072 | 逻辑上下文（KVMem 把它"摊"到主机内存） |
| `--kvmem-budget` / `--kvmem-gen-reserve` | 24576 / 10240 | GPU 常驻窗口：24K 检索 + 10K 生成；**单轮输出不能超过 reserve** |
| `--kv-dtype` | `q8_0` | 30 系没有 FP8 张量核；q8_0 是这条线的默认档 |
| `--kvmem-block-tokens` | 128 | 检索块大小（budget/reserve 必须是它的整数倍） |
| `--flash-attn` | on | |
| `--reasoning-budget` | 4096 | 思考预算；请求级 `chat_template_kwargs.enable_thinking=false` 可关 |

**MTP（可选，+15~22% decode）**：头部来自社区 `ProCreations/Ternary-Bonsai-2-27B-MTP`
（固定 revision，非官方权重），按官方流程量化合并：

```bash
# 合并脚本只需要 numpy（不需要 pip/root）：wheel 本身就是 zip，解压后用 PYTHONPATH 指过去
curl -O https://pypi.tuna.tsinghua.edu.cn/simple/numpy/numpy-2.5.3-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
unzip -oq numpy-*.whl -d pylibs/

PYTHONPATH=$PWD/pylibs python3 scripts/bonsai-mtp-graft.py \
  --base models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --head model_mtp.safetensors \
  --output models/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf --head-type Q4_0
# 脚本自带 SHA256 校验（base 53107f53…、head 7a4a18b2…），输出 6.19 GB
```

> ⚠️ MTP 头只在 `--spec-type draft-mtp` 真正生效时才被加载；否则日志会出现
> `model has unused tensor blk.64.nextn.*` 的告警（无害，说明头没被用上）。
> 本机实测 **MTP1 是甜点，MTP2 反而变慢**（34.6 vs 39.3 t/s）——3060 算力/带宽都低，
> 多草稿的额外计算吃掉了收益。

### 接 OpenAI 客户端（ZCode / Chatbox / Cherry Studio 等）

| 客户端里填 | 值 |
|---|---|
| Base URL | `http://127.0.0.1:18202/v1` |
| API Key | 任意（服务端不校验，如 `ssss`） |
| 模型名 | 任意（如 `bonsai`）——服务端忽略该字段 |

**★ 必打兼容补丁：`reasoning_effort` 别名。** 模型自带聊天模板只认
`xhigh / medium / low`，而 OpenAI 系客户端默认发 **`high`**（ZCode 发
`{"reasoning_effort":"high","enable_thinking":true}`），模板会直接抛异常 → **HTTP 400**：

```
Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.
```

修法与 ninfer 线的 `patches/apply-compat.sh`（P1）同类，但**不用重编引擎**——挂一份模板副本即可：

```bash
./patches/apply-kvmem-compat.sh models/Ternary-Bonsai-2-27B-PTQ1_0.gguf   # 生成别名模板
# start-kvmem.sh 会自动带上 --chat-template-file patches/bonsai-chat-template-alias.jinja
```

**实测接 ZCode**（把本服务配成 provider，2026-10-03）：

| 项 | 实测 |
|---|---|
| 请求规模 | 系统提示 + **33 个工具** = **22,148 token** 提示 |
| 首轮（冷 prefill） | **~100 s**（prefill 231.6 t/s） |
| 第二轮起 | **1~2 s**（KVMem 前缀缓存命中 22147/22148 token） |
| 单轮生成上限 | 10240 token（= `--kvmem-gen-reserve`）；客户端要 32000 会被截断（`finish_reason: length`） |
| 工具调用 | 文本回复正常；27B 三元模型对 33 工具的调用能力**未经充分验证**，别期待能稳定当编码 agent |

> 一句话：接上去能用，**首轮慢、之后每轮 1~2 秒**；想让它当 agent 干活，建议先把
> ZCode 的思考关掉或调低（思考 token 会计入 10240 上限，且明显拖慢）。

---

## 6. 与 ninfer 线（报告里的"路线 B"）怎么选

同一张 3060 12GB 上，两条线是**速度 vs 上下文**的取舍（ninfer 数据引自
`汇总-报告与技能/报告/调优报告2-DFlash2路线-104tps.md`，本仓库未复测）：

| | 路线 A：本页（llama.cpp + KVMem） | 路线 B：自编 ninfer sm_86 |
|---|---|---|
| decode | 34 t/s（MTP1: 39~41） | **108.7 t/s**（DFlash2 K=7，实测） |
| prefill | 206~233 t/s | 640 t/s |
| 上下文 | **128K**（109k 召回已验证） | 24K（其实测最终配置；调优单估算 int8 可到 64K，未实测） |
| 峰值显存 | 8.0~8.8 GiB | 约 8.2 GiB（6.70 权重 + 1.51 运行时，int8/32K） |
| 引擎改动 | 无需改源码（官方分支，只改编译架构） | 需改 4 处源码 + 补 FP8 stub |
| 引擎来源 | 公开仓库、可自编 | 交付包源码树（需自编） |

**一句话**：要**长上下文/多轮记忆/低显存**选 A；要**出字快**（对话、代码补全这类吃速度的场景）选 B。
两条线**互斥**（都吃满显存），`start-kvmem.sh` 与 `start-ninfer.sh` 不要同时起。

---

## 7. 本机踩到的坑（都已解决）

| # | 症状 | 原因 / 处置 |
|---|---|---|
| 1 | `nvvm/bin/cicc: not found` | CUDA 13 的 nvcc 需要单独的 `libnvvm-13-2` 包（apt 依赖列表里容易漏）→ 补包重解 |
| 2 | `--ui-dir must contain a readable index.html` | 源码树 `ui/` 是 Svelte **源码**不是静态页（要 Node 22 构建）→ 无 UI 时不要传 `--ui-dir` |
| 3 | 启动成功但日志 `spec=off`、`unused tensor blk.64.nextn.*` | MTP 头没启用：`--spec-type` 被写成了 `none`（脚本里做字符串替换时数组元素是分开的）→ 用独立变量拼 `SPEC_ARGS` |
| 4 | 走系统代理下载 6 GB 模型只有 ~390 KB/s | 国内镜像应**绕过代理直连**：`curl --noproxy '*'` → hf-mirror 实测 35 MB/s |
| 5 | `python3` 无 pip / 无 numpy（合并 MTP 头要用） | 不要装系统包：直接下 `numpy-*-cp314-*-manylinux*.whl`（wheel 就是 zip），`unzip` 到目录 + `PYTHONPATH` 即可 |
| 6 | 长 prompt 下 decode 偏低（27 t/s） | 短输出样本（8 token）被每轮固定开销主导 —— **测速要看 ≥300 token 的长输出**（报告里"短输出伪造低接受率"的同源教训） |
| 7 | 接 ZCode/客户端后整轮 400：`Unexpected reasoning effort high` | 客户端发 `reasoning_effort:"high"`，模板只认 xhigh/medium/low，模板抛异常 ⇒ 挂别名模板（`patches/apply-kvmem-compat.sh` + `--chat-template-file`），**不用重编**。注意：关掉思考时该字段不校验，所以只有"思考开"的客户端才会踩到 |
| 8 | 长会话某轮开始稳定 400：`{"error":"prompt + max_tokens exceeds n_ctx"}` | **上下文用满**：提示 token + 单轮预留（10240）> `-c`(131072)，上限其实是 **120832** 个提示 token。这是校验期秒拒（不是超时）。处置：新开会话；或 `KVMEM_RESERVE=4096`（让出约 6K 提示空间，输出上限同步降）；或 `KVMEM_CTX=262144` 顶到模型原生 256K（KV 在主机内存，显存不变，超长召回属官方实验特性）|

---

## 8. 复现清单（TL;DR）

```bash
# 1) 引擎（免 root，见 build-kvmem-sm86.sh 全文）
./scripts/build-kvmem-sm86.sh                       # 解包 CUDA → clone 分支 → apply-patches → ninja

# 2) 模型（走国内镜像，直连不走代理；sha256 53107f530aa52eb0…）
curl -L --noproxy '*' -o models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  https://hf-mirror.com/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf

# 3) 起服务（128K）
./scripts/start-kvmem.sh work                       # → http://127.0.0.1:18202

# 4) 冒烟
curl -s --noproxy '*' http://127.0.0.1:18202/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"用一句话中文自我介绍。"}],"max_tokens":64,"temperature":0}'

# 5) 基准（本表全部数字由它产出）
python3 scripts/bench_kvmem.py
```

许可与出处：引擎 `kvmem-llama.cpp` / PrismML 的 `llama.cpp` fork（均在各自目录保留 LICENSE）；
模型 `prism-ml/Ternary-Bonsai-2-27B-gguf`（Apache-2.0，"Created using Bonsai by Prism ML."）；
MTP 头为社区产物（`ProCreations`），非官方权重。本页数据为本机实测，单卡单样本，不代表上限。
