#!/usr/bin/env bash
# =============================================================================
# Route A「省显存档」· llama.cpp + KVMem 启动脚本
# 目标硬件：RTX 3060 12GB（compute capability 8.6 / sm_86 / Ubuntu）
#
# 与 ninfer 线（scripts/start-ninfer.sh）互斥 —— 两条线都要吃满显存，勿同时启动。
#
# 用法:
#   ./start-kvmem.sh [profile] [port]       启动（默认 profile=work 端口=18202）
#   ./start-kvmem.sh stop   [profile]       停止
#   ./start-kvmem.sh status [profile]       状态
#   ./start-kvmem.sh logs   [profile]       跟踪日志#
# profile（3060 12GB 实测）:
#   work   默认。128K 逻辑上下文（KVMem 历史 KV 放主机内存，GPU 只驻留窗口），
#          GPU 窗口 24K 检索 + 10K 生成预留，主 KV q8_0，思考开（预算 4096）。
#   low    省显存档：64K 上下文、12K 窗口 + 6K 预留（边办公边跑）。
#   fast   短上下文快速档：8K 上下文、4K 窗口 + 2K 预留。
#   long   256K 档（= 模型原生上限，2026-10-08 实机验证）：给长会话用。
#          实测：12.1 万 token 提示预填充 571 s（211 t/s），显存 8.6 GiB（与 128K 档几乎相同，
#          历史 KV 在主机内存），主机内存 15 GiB/30 GiB。提示上限 = ctx − reserve = 251904。
#
# 环境变量覆盖（长会话撞上下文上限时用）:
#   KVMEM_CTX=262144     逻辑上下文顶到模型原生 256K（历史 KV 在内存，显存不变）
#   KVMEM_RESERVE=4096   缩小单轮生成预留 → 给提示让出约 6K 空间（输出上限同步降到 4096）
#   KVMEM_BUDGET=32768   加大 GPU 检索窗口（多吃显存）
# 例：KVMEM_CTX=262144 KVMEM_RESERVE=8192 ./scripts/start-kvmem.sh work
#
# 首次使用前先跑 scripts/build-kvmem-sm86.sh 准备引擎（免 root 工具链 + 编译）。
#
# 请求方式（OpenAI 兼容，与 ninfer 线一致；注意 30 系没有 FP8 张量核，
# 该线用 q8_0 KV，不是 fp8）:
#   curl -s http://127.0.0.1:18202/v1/models
#   curl -s http://127.0.0.1:18202/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"model":"bonsai","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
#   Web UI: http://127.0.0.1:18202/
# =============================================================================
set -uo pipefail

# ---- 路径（按你的机器改）-----------------------------------------------------
KVROOT="${KVROOT:-/mnt/D/code/kvmem-bonsai}"     # 引擎源码树 + 免 root CUDA 工具链
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${KVMEM_MODEL:-$REPO/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf}"
# MTP 投机解码（可选，+30% 左右 decode）：KVMEM_MTP=1 时换用 graft 过的 MTP 模型
if [[ "${KVMEM_MTP:-0}" == "1" ]]; then
    MODEL="${KVMEM_MODEL:-$REPO/models/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf}"
    DRAFT_N="${KVMEM_DRAFT_N:-1}"       # 1 或 2（2 更快，草稿命中率略降）
fi
BIN="$KVROOT/kvmem-src/build/bin/llama-kvmem-server"
CUDA_LIB="$KVROOT/cuda-stage/usr/local/cuda-13.2/lib64"
UI_DIR="$KVROOT/kvmem-src/ui/lightweight"
LOGDIR="${LOGDIR:-$HOME/kvmem-logs}"

# ---- 档位 --------------------------------------------------------------------
PROFILE="${1:-work}"
ACTION="start"
if [[ "${1:-}" == "stop" || "${1:-}" == "status" || "${1:-}" == "logs" ]]; then
    ACTION="$1"; PROFILE="${2:-work}"; PORT="${3:-18202}"
else
    PORT="${2:-18202}"
fi

case "$PROFILE" in
    work) CTX=131072; BUDGET=24576; RESERVE=10240; KV="q8_0" ;;  # 128K 逻辑上下文
    low)  CTX=65536;  BUDGET=12288; RESERVE=6144;  KV="q8_0" ;;  # 64K / 省显存
    fast) CTX=8192;   BUDGET=4096;  RESERVE=2048;  KV="q8_0" ;;  # 8K / 最省显存
    long) CTX=262144; BUDGET=24576; RESERVE=10240; KV="q8_0" ;;  # 256K：模型原生上限，长会话用
    *) echo "未知档位: $PROFILE（可选 work|low|fast|long）" >&2; exit 2 ;;
esac

# 可用环境变量覆盖（例：上下文顶到模型原生 256K；或缩小单轮预留给长提示让路）
#   KVMEM_CTX=262144 KVMEM_RESERVE=8192 ./scripts/start-kvmem.sh work
# 约束：budget + reserve ≤ ctx，且 reserve ≥ 客户端的单轮输出上限
CTX="${KVMEM_CTX:-$CTX}"
RESERVE="${KVMEM_RESERVE:-$RESERVE}"
BUDGET="${KVMEM_BUDGET:-$BUDGET}"
if (( BUDGET + RESERVE > CTX )); then
    echo "参数不合法: budget($BUDGET) + reserve($RESERVE) > ctx($CTX)" >&2; exit 2
fi

PIDFILE="$LOGDIR/kvmem-$PROFILE.pid"
LOGFILE="$LOGDIR/kvmem-$PROFILE.log"
mkdir -p "$LOGDIR"

case "$ACTION" in
stop)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        kill "$(cat "$PIDFILE")" && echo "已停止 profile=$PROFILE (pid $(cat "$PIDFILE"))"
        rm -f "$PIDFILE"
    else
        # 兜底：按可执行名找（不要用 pkill -f，模式会匹配到脚本自身命令行）
        if pgrep -x llama-kvmem-ser >/dev/null 2>&1; then
            pkill -x llama-kvmem-ser && echo "已停止（按进程名）"
        else
            echo "未在运行 profile=$PROFILE"
        fi
    fi
    exit 0
    ;;
status)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "运行中: profile=$PROFILE pid=$(cat "$PIDFILE") port=$PORT"
        curl -s --noproxy '*' -m 5 "http://127.0.0.1:$PORT/v1/models" | head -c 300; echo
    else
        echo "未在运行 profile=$PROFILE"
    fi
    exit 0
    ;;
logs)
    exec tail -f "$LOGFILE"
    ;;
esac

[[ -x "$BIN"  ]] || { echo "缺引擎: $BIN（先跑 scripts/build-kvmem-sm86.sh）" >&2; exit 1; }
[[ -r "$MODEL" ]] || { echo "缺模型: $MODEL" >&2; exit 1; }

# 免 root CUDA 运行库（编译产物动态链接到 KVROOT 内的 CUDA 13.2）
export LD_LIBRARY_PATH="$CUDA_LIB:$KVROOT/cuda-stage/usr/local/cuda-13.2/nvvm/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

SPEC_ARGS=(--spec-type none)
if [[ "${KVMEM_MTP:-0}" == "1" ]]; then
    # 把 --spec-type none 换成 MTP；草稿 KV 用 f16（官方默认），草稿窗口= reserve 内
    SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "${DRAFT_N:-1}" --spec-kv-dtype f16)
fi

ARGS=(-m "$MODEL" -ngl 99 --host 127.0.0.1 --port "$PORT"
      -c "$CTX" -b 128 --ubatch-size 128 -n "$RESERVE"
      --kvmem-budget "$BUDGET" --kvmem-gen-reserve "$RESERVE" --kvmem-block-tokens 128
      --kv-dtype "$KV" "${SPEC_ARGS[@]}" --kvmem-mtp-state snapshots
      --kvmem-query-policy user --kvmem-query-replay auto --flash-attn on
      --enable-thinking --reasoning-budget 4096)
# Web UI 是可选件：源码树里的 ui/ 是 Svelte 源（需 Node 构建），
# 预编译包里的 share/kvmem/ui 才是静态页；两者都没有时只提供 API。
if [[ -r "$UI_DIR/index.html" ]]; then ARGS+=(--ui-dir "$UI_DIR"); fi

# 对话模板别名补丁：模型自带模板只认 reasoning_effort ∈ {xhigh, medium, low}，
# 而很多客户端会发 high / max / minimal（ZCode 就发 high）→ 模板抛异常 → HTTP 400。
# 挂一份把别名归一化的模板副本（原模板 + 归一逻辑），不重编引擎。
# 生成/重生成：patches/apply-kvmem-compat.sh <model.gguf>
TEMPLATE="$REPO/patches/bonsai-chat-template-alias.jinja"
if [[ -r "$TEMPLATE" ]]; then ARGS+=(--chat-template-file "$TEMPLATE"); fi

echo "启动 llama-kvmem-server | profile=$PROFILE | 端口=$PORT | 上下文=$CTX | GPU 窗口=$BUDGET+$RESERVE | KV=$KV"
echo "日志: $LOGFILE"
nohup "$BIN" "${ARGS[@]}" >"$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

# 等待就绪（模型 6 GB，首次加载约 10-30 s）
for i in $(seq 1 90); do
    if curl -s --noproxy '*' -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "就绪: http://127.0.0.1:$PORT/（Web UI 同址）"
        exit 0
    fi
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "进程已退出，看日志:"; tail -20 "$LOGFILE"; exit 1; }
    sleep 2
done
echo "90 次探测未就绪（可能仍在加载），看日志: $LOGFILE"
