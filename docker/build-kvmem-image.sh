#!/usr/bin/env bash
# =============================================================================
# 构建 3060（sm_86）容器镜像 —— llama.cpp + KVMem 线（路线 A「省显存档」）
#
# 用法:
#   ./docker/build-kvmem-image.sh                       # 默认 arch 86 + CUDA 13.3.0
#   ./docker/build-kvmem-image.sh -a 89                 # 89/120a 同分支官方编过，本机未验证，属尝鲜
#   ./docker/build-kvmem-image.sh -b nvidia/cuda:<13.2.x>-devel-ubuntu24.04 # 贴合作者工具链基线（tag 以 Docker Hub 为准）
#   ./docker/build-kvmem-image.sh -m 1                  # 国内构建（apt 走清华源）
#   ./docker/build-kvmem-image.sh -j 3                  # nvcc 并行度（默认 6；runner 4 核建议 3）
#
# 前置:
#   · docker 可用（不在 docker 组时脚本自动改用 sudo）
#   · 首次构建会拉取 CUDA devel 基础镜像（约 6~7 GB）
#   · 构建期需要访问 GitHub（克隆 kvmem-llama.cpp 与 llama.cpp 子模块，revision 已 pin）
# 产物: llama-kvmem-bonsai:cuda<版本>-arch<架构>（默认 cuda13.3-arch86）
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCH=86
BASE=nvidia/cuda:13.3.0-devel-ubuntu24.04
JOBS="${KVMEM_JOBS:-6}"
MIRROR=0
TAG=""
while getopts "a:b:j:m:t:" o; do case "$o" in
    a) ARCH="$OPTARG" ;;
    b) BASE="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    m) MIRROR="$OPTARG" ;;
    t) TAG="$OPTARG" ;;
    *) grep '^#' "$0" | sed -n '2,16p'; exit 1 ;;
esac; done

[[ "$ARCH" =~ ^(86|89|120a)$ ]] || { echo "错误: -a 只能是 86（RTX 30 系）/ 89（40 系）/ 120a（50 系）" >&2; exit 1; }
command -v docker >/dev/null || { echo "错误: 需要 docker" >&2; exit 1; }
CUDASHORT="$(echo "$BASE" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?$' | cut -d. -f1,2 || true)"
[[ -n "$CUDASHORT" ]] || { echo "错误: -b 需为 nvidia/cuda:<版本> 形式" >&2; exit 1; }
TAG="${TAG:-llama-kvmem-bonsai:cuda${CUDASHORT}-arch${ARCH}}"

DOCKER="docker"; docker info >/dev/null 2>&1 || DOCKER="sudo docker"

# 只把需要的文件放进构建上下文（避免把 8 GB 的 models/ 整个塞进去）
CTX="$(mktemp -d /tmp/kvmem-docker-ctx.XXXXXX)"
trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/docker/kvmem" "$CTX/scripts" "$CTX/patches"
cp "$ROOT/docker/kvmem/Dockerfile" "$ROOT/docker/kvmem/docker-entrypoint.sh" "$CTX/docker/kvmem/"
cp "$ROOT/scripts/kvmem-proxy.py" "$CTX/scripts/"
cp "$ROOT/patches/bonsai-chat-template-alias.jinja" "$CTX/patches/"

echo ">> docker build（${BASE} , arch=${ARCH} , 并行=${JOBS} , 国内源=${MIRROR}）→ $TAG"
$DOCKER build \
    --build-arg CUDA_BASE="$BASE" \
    --build-arg CUDA_ARCH="$ARCH" \
    --build-arg KVMEM_JOBS="$JOBS" \
    --build-arg USE_MIRROR="$MIRROR" \
    -t "$TAG" \
    "$CTX"

echo ">> 构建完成: $TAG"
echo "   起服务:   docker compose -f docker/kvmem/docker-compose.kvmem.yml up -d"
echo "   或裸跑:   $DOCKER run --rm -it --gpus all -p 127.0.0.1:18210:18210 -p 127.0.0.1:18202:18202 \\"
echo "               -v $ROOT/models:/models:ro $TAG"
