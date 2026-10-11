# 3060（sm_86）容器镜像 —— llama.cpp + KVMem 线

> 给 **RTX 30 系（sm_86）**用的镜像，和 `docker/` 下那套 ninfer 镜像（**sm_89 / sm_120a = 40/50 系**）
> 是**两条不同的线**，互不影响：
>
> | | 本镜像（`docker/kvmem/`） | ninfer 镜像（`docker/`） |
> |---|---|---|
> | 引擎 | `kvmem/kvmem-llama.cpp` Bonsai 分支（三元内核 + KVMem） | ninfer（交付包引擎树） |
> | 架构 | **sm_86（RTX 30 系）**；同分支官方也编过 89/120a（本机未验证，可用 `-a` 试） | sm_89 / sm_120a |
> | 上下文 | **128K~256K**（历史 KV 在主机内存，GPU 只驻留窗口） | 24K~112K（12GB 卡） |
> | 出字 | ~34 t/s（MTP1 39~41） | ~109 t/s |
> | 内置 | 视觉塔开关、MTP 开关、**上下文自动压缩代理**、客户端兼容模板 | — |

## 1. 构建

```bash
./docker/build-kvmem-image.sh                  # 默认 arch 86 + CUDA 13.3.0 → llama-kvmem-bonsai:cuda13.3-arch86
./docker/build-kvmem-image.sh -m 1             # 国内构建（apt 走清华源）
./docker/build-kvmem-image.sh -j 3             # nvcc 并行度（默认 6）
./docker/build-kvmem-image.sh -b nvidia/cuda:<13.2.x>-devel-ubuntu24.04   # 贴合作者工具链基线（tag 以 Docker Hub 为准）
```

- **国内构建建议**：`./docker/build-kvmem-image.sh -m 1 -n host`
  —— `-m 1` 让容器内 apt 走清华源；`-n host` 让**构建容器与宿主同网络**，这样宿主机代理
  （如 `127.0.0.1:10808`）才可达、克隆 GitHub 源码才稳（脚本会自动透传 `HTTPS_PROXY`，也可 `-p` 显式指定）。
  docker daemon 侧建议配 registry mirror 加速拉基础镜像。
- **本机已实测**：`-a 86 -n host -m 1` 构建通过 → `llama-kvmem-bonsai:cuda13.3-arch86`，**2.25 GB**；
  镜像在 `--gpus all` 下引擎能加载运行，入口脚本缺模型时给出明确报错。
- 构建期会**克隆** `kvmem-llama.cpp`（分支 `kvmem-bonsai-llama.cpp`）与其 `llama.cpp` 子模块，
  revision 已 pin 在 Dockerfile 里（可复现）；需能访问 GitHub。
- 首次约 **10~25 分钟**（含拉取 CUDA devel 基础镜像 ~7 GB）；产物镜像约 **2~4 GB**。
- CI 也会自动构建并推送：`.github/workflows/docker-build-kvmem.yml` → `ghcr.io/<owner>/llama-kvmem-bonsai`。
  首次拉包前要在 GitHub 网页把该包设为 Public（或在本地 `docker login ghcr.io`）。

## 2. 放模型（镜像不含权重）

把下面文件放进**仓库的 `models/`**（compose 会只读挂载到容器 `/models`）：

| 文件 | 必需 | 说明 |
|---|---|---|
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf`（5.95 GB） | ✅ | 主模型（魔搭/HF：`prism-ml/Ternary-Bonsai-2-27B-gguf`） |
| `Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf`（6.19 GB） | 可选 | 合并了 MTP 头 → 出字 +15~22%（`patches/` 里有 graft 脚本说明） |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`（0.63 GB） | 可选 | 视觉塔 → 能贴图 |

**文件在同目录就自动启用**（`KVMEM_MTP=0` / `KVMEM_VISION=0` 可分别关掉）。

## 3. 运行

```bash
docker compose -f docker/kvmem/docker-compose.kvmem.yml up -d
docker compose -f docker/kvmem/docker-compose.kvmem.yml logs -f      # 首次加载约 10~30 s
```

客户端（ZCode / Chatbox / Cherry Studio 等）填：

| 项 | 值 |
|---|---|
| Base URL | **`http://127.0.0.1:18210/v1`**（走压缩代理；`18202` 是引擎直连，调试用） |
| API Key | 任意 |
| 模型名 | 任意 |

裸跑（不用 compose）：

```bash
docker run --rm -it --gpus all \
  -p 127.0.0.1:18210:18210 -p 127.0.0.1:18202:18202 \
  -v "$PWD/models:/models:ro" \
  llama-kvmem-bonsai:cuda13.3-arch86
```

## 4. 常用环境变量（与宿主机 `scripts/start-kvmem.sh` 同名同义）

| 变量 | 默认 | 说明 |
|---|---|---|
| `KVMEM_CTX` | 131072 | 逻辑上下文；长会话用 `262144`（KV 在主机内存，显存不变） |
| `KVMEM_RESERVE` | 10240 | 单轮输出预留 = **单轮输出上限**（含思考 token） |
| `KVMEM_BUDGET` | 24576 | GPU 检索窗口 |
| `KVMEM_KV` | q8_0 | KV 量化档（**30 系没有 FP8 张量核，不要 fp8**） |
| `KVMEM_MTP` / `KVMEM_DRAFT_N` | 1 / 1 | MTP 投机；`DRAFT_N=2` 在 3060 上实测更慢 |
| `KVMEM_VISION` / `KVMEM_VISION_CPU` | 1 / 0 | 视觉塔；`CPU=1` 编码器放 CPU 省显存 |
| `KVMEM_PROXY` / `KVMEM_PROXY_THRESHOLD` | 1 / 45000 | 上下文自动压缩代理与其触发阈值 |

> ⚠️ **提示长度硬上限** = `KVMEM_CTX − KVMEM_RESERVE`（131072−10240 = 120832）。
> 超了引擎会**秒拒** `prompt + max_tokens exceeds n_ctx`；压缩代理会在这之前先把早期对话摘要掉。

## 5. 排错

| 症状 | 处置 |
|---|---|
| 起容器就退，日志 `缺模型: /models/…` | 模型没挂对；确认 `models/*.gguf` 在仓库 `models/` 下，或自己 `-v <目录>:/models:ro` |
| `docker: Error response … could not select device driver ""` | 宿主没装 nvidia-container-toolkit |
| 客户端整轮 400 `Unexpected reasoning effort high` | 镜像自带兼容模板（`KVMEM_TEMPLATE`），若被覆盖请检查该文件是否存在 |
| 客户端整轮 400 `System message must be at the beginning` | 同上（兼容模板的补丁 2） |
| 看着"没输出"、日志在跑 `prompt processing … progress=` | 在预填充，等它跑完；**中途不要重发**（会从头开始） |
| 出字一块一块蹦 | 压缩代理的 SSE 转发必须用 `read1`（镜像内已修；改代码时别改回落 `read`） |

引擎与代理日志：`docker compose … logs -f`（代理的摘要状态在 `kvmem-proxy-state` 卷里，
删卷即重置记忆）。
