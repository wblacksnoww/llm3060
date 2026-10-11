#!/usr/bin/env bash
# =============================================================================
# 容器入口（3060 线）—— 起 llama-kvmem-server（+ 可选上下文压缩代理）
#
# 环境变量与宿主机 scripts/start-kvmem.sh 对齐：
#   KVMEM_MODEL=/models/xxx.gguf     模型路径（默认 /models 下的 PTQ1_0）
#   KVMEM_CTX=131072                 逻辑上下文（长会话用 262144）
#   KVMEM_RESERVE=10240              单轮输出预留（= 单轮输出上限，含思考）
#   KVMEM_BUDGET=24576               GPU 检索窗口
#   KVMEM_KV=q8_0                    KV 量化档（30 系没有 FP8 张量核，别用 fp8）
#   KVMEM_MTP=1                      有 MTP 模型就自动开投机（0 关）
#   KVMEM_VISION=1                   有视觉塔就自动开（KVMEM_VISION_CPU=1 编码器放 CPU）
#   KVMEM_PROXY=1                    起上下文自动压缩代理（0 关）
#   KVMEM_PROXY_THRESHOLD=45000      代理触发压缩的估算 token 阈值
#   KVMEM_ENGINE_PORT=18202 / KVMEM_PROXY_PORT=18210
# =============================================================================
set -euo pipefail

: "${KVMEM_MODEL:=/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf}"
: "${KVMEM_CTX:=131072}"
: "${KVMEM_RESERVE:=10240}"
: "${KVMEM_BUDGET:=24576}"
: "${KVMEM_KV:=q8_0}"
: "${KVMEM_ENGINE_PORT:=18202}"
: "${KVMEM_PROXY_PORT:=18210}"
: "${KVMEM_REASONING_BUDGET:=4096}"

BIN=/opt/kvmem/bin/llama-kvmem-server
MODEL_DIR="$(dirname "$KVMEM_MODEL")"

[[ -x "$BIN" ]] || { echo "镜像内缺引擎: $BIN" >&2; exit 1; }
[[ -r "$KVMEM_MODEL" ]] || {
    echo "缺模型: $KVMEM_MODEL" >&2
    echo "把 .gguf 放到宿主机目录并挂载到 /models（compose 默认挂 ../models，只读）" >&2
    exit 1
}

# ---- 可选件：MTP 投机（+15~22% 出字）与视觉塔，都是「文件在就自动开」 ----
MTP="$MODEL_DIR/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf"
MMPROJ="$MODEL_DIR/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"

SPEC=(--spec-type none); MTP_STATE=关
if [[ "${KVMEM_MTP:-1}" != "0" && -r "$MTP" ]]; then
    SPEC=(--spec-type draft-mtp --spec-draft-n-max "${KVMEM_DRAFT_N:-1}" --spec-kv-dtype f16)
    MTP_STATE="开(MTP${KVMEM_DRAFT_N:-1})"
fi

VISION=(); VISION_STATE=关
if [[ "${KVMEM_VISION:-1}" != "0" && -r "$MMPROJ" ]]; then
    VISION=(--mmproj "$MMPROJ")
    [[ "${KVMEM_VISION_CPU:-0}" == "1" ]] && VISION+=(--no-mmproj-offload)
    VISION_STATE=开
fi

TEMPLATE=()
[[ -r "${KVMEM_TEMPLATE:-}" ]] && TEMPLATE=(--chat-template-file "$KVMEM_TEMPLATE")

# ---- 上下文自动压缩代理（默认开）：客户端接 $KVMEM_PROXY_PORT ----
if [[ "${KVMEM_PROXY:-1}" != "0" ]]; then
    KVMEM_PROXY_PORT="$KVMEM_PROXY_PORT" \
    KVMEM_PROXY_UPSTREAM_PORT="$KVMEM_ENGINE_PORT" \
    KVMEM_PROXY_THRESHOLD="${KVMEM_PROXY_THRESHOLD:-45000}" \
    python3 /opt/kvmem/kvmem-proxy.py &
fi

echo "llama-kvmem-server 启动 | 引擎 :$KVMEM_ENGINE_PORT ｜ 压缩代理 :$KVMEM_PROXY_PORT ｜ 上下文=$KVMEM_CTX ｜ GPU 窗口=$KVMEM_BUDGET+$KVMEM_RESERVE ｜ KV=$KVMEM_KV ｜ 视觉=$VISION_STATE ｜ 投机=$MTP_STATE"

exec "$BIN" \
    -m "$KVMEM_MODEL" -ngl 99 --host 0.0.0.0 --port "$KVMEM_ENGINE_PORT" \
    -c "$KVMEM_CTX" -b 128 --ubatch-size 128 -n "$KVMEM_RESERVE" \
    --kvmem-budget "$KVMEM_BUDGET" --kvmem-gen-reserve "$KVMEM_RESERVE" --kvmem-block-tokens 128 \
    --kv-dtype "$KVMEM_KV" "${SPEC[@]}" --kvmem-mtp-state snapshots \
    --kvmem-query-policy user --kvmem-query-replay auto --flash-attn on \
    --enable-thinking --reasoning-budget "$KVMEM_REASONING_BUDGET" \
    "${VISION[@]}" "${TEMPLATE[@]}"
