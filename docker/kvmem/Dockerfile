# =============================================================================
# 3060 / RTX 30 系（sm_86）容器镜像 —— llama.cpp + KVMem 线（路线 A「省显存档」）
#
# 与 docker/Dockerfile（ninfer 线，sm_89 / sm_120a）的区别：
#   * 引擎不同：本镜像编译 kvmem/kvmem-llama.cpp 的 Bonsai 分支（三元内核 + KVMem 分层 KV）
#   * 默认架构 sm_86（RTX 30 系）；同一分支官方也编过 89 / 120a，可用 -a 覆盖
#   * 不含模型制品：模型单独下载、运行时挂载（与 ninfer 线同设计）
#   * 内置「上下文自动压缩代理」与客户端兼容模板，起容器即用
#
# 源码 revision 已 pin（= 本机已验证部署的那一份）：
#   kvmem-llama.cpp 974258ad…（kvmem-bonsai-llama.cpp 分支）· llama.cpp 子模块 9a9394a8…
#   （子模块 revision 若与 pin 不同会打一行提示，不中断构建）
#
# CUDA 基线：作者验证的是 13.2 Update 2（nvcc 13.2.86）；本文件默认 13.3.0
#   （仓库 CI 一直在用的 tag，满足"13.2 Update 2 或更新"）。
#   想完全贴合作者工具链：用 -b 指定 13.2.x 的 devel tag（具体 tag 以 Docker Hub 实际存在的为准）
#
# 用法：./docker/build-kvmem-image.sh   （构造精确的构建上下文与参数，见该脚本）
# =============================================================================
ARG CUDA_BASE=nvidia/cuda:13.3.0-devel-ubuntu24.04
FROM ${CUDA_BASE} AS build

ARG CUDA_ARCH=86
ARG KVMEM_JOBS=6
ARG KVMEM_REPO=https://github.com/kvmem/kvmem-llama.cpp.git
ARG KVMEM_REV=974258ad2956ff921419c8218b5075db3a3a278d
ARG LLAMA_REV=9a9394a895b96003ca842a6041cb28ac49a108f7
# 1 = 国内构建：apt 换清华源并禁用镜像内 CUDA apt 仓库（编译器已随 devel 镜像内置）
ARG USE_MIRROR=0

RUN set -eux; \
    if [ "$USE_MIRROR" = "1" ]; then \
        rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/cuda*.sources; \
        for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; do \
            [ -f "$f" ] || continue; \
            sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' "$f"; \
        done; \
    fi; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        build-essential git ca-certificates cmake ninja-build libssl-dev pkg-config; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
# 部分克隆 + 按 revision 检出：可复现（不随分支前进漂移），也不需要完整历史
RUN git clone --filter=blob:none --no-checkout "$KVMEM_REPO" /src \
 && git -C /src checkout --quiet "$KVMEM_REV" \
 && git -C /src submodule update --init --recursive --depth 1 \
 && ( [ "$(git -C /src/llama.cpp rev-parse HEAD)" = "$LLAMA_REV" ] \
      || echo "注意: llama.cpp 子模块 revision 与 pin($LLAMA_REV) 不同（上游可能已前进）" )

RUN ./scripts/apply-patches.sh

# 与本机验证过的编译命令逐条一致（arch 用 <N>-real，只出真实架构、不生成 PTX）
RUN cmake -S /src -B /src/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}-real" \
        -DGGML_CUDA=ON \
        -DKVMEM_BUILD_LLAMA=ON \
        -DLLAMA_KVMEM=ON \
        -DLLAMA_KVMEM_ROOT=/src

RUN ninja -C /src/build -j"${KVMEM_JOBS}" llama-kvmem-server llama-kvmem-cli

# ------------------------------ 运行时（瘦身）------------------------------
# 用官方 ubuntu 镜像 + 从构建阶段拷 CUDA 运行库（libcudart / libcublas），
# 驱动 libcuda.so.1 由宿主经 nvidia-container-toolkit 注入（--gpus all / compose devices）
FROM ubuntu:24.04 AS runtime

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      libstdc++6 libgomp1 libssl3t64 zlib1g libzstd1 python3 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13*   /opt/cuda/lib/
COPY --from=build /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.13*   /opt/cuda/lib/
COPY --from=build /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.13* /opt/cuda/lib/
COPY --from=build /src/build/bin/ /opt/kvmem/bin/
COPY docker/kvmem/docker-entrypoint.sh /opt/kvmem/docker-entrypoint.sh
COPY scripts/kvmem-proxy.py /opt/kvmem/kvmem-proxy.py
COPY patches/bonsai-chat-template-alias.jinja /opt/kvmem/bonsai-chat-template-alias.jinja
RUN chmod +x /opt/kvmem/docker-entrypoint.sh

ENV LD_LIBRARY_PATH=/opt/cuda/lib:/opt/kvmem/bin \
    KVMEM_MODEL=/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    KVMEM_TEMPLATE=/opt/kvmem/bonsai-chat-template-alias.jinja \
    KVMEM_ENGINE_PORT=18202 \
    KVMEM_PROXY_PORT=18210

# 18202 = 引擎直连；18210 = 上下文压缩代理（客户端应接这个）
EXPOSE 18202 18210

ENTRYPOINT ["/opt/kvmem/docker-entrypoint.sh"]
