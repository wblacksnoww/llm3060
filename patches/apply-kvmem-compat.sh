#!/usr/bin/env bash
# =============================================================================
# KVMem 线（路线 A）兼容补丁：把模型自带聊天模板改成"客户端友好"版本
#
# 用不重编引擎的方式（挂 --chat-template-file）修两类真实踩到的 400：
#
# 【补丁 1】reasoning_effort 别名归一
#   模型模板只接受 xhigh/medium/low，而 OpenAI 系客户端默认发 "high"
#   （ZCode 发 {"reasoning_effort":"high","enable_thinking":true}）→ 模板抛异常：
#     Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh...
#   映射：high/max/maximum/highest→xhigh；minimal/min/lowest→low；未知→xhigh。
#   （与 ninfer 线 patches/apply-compat.sh 的 P1 同类问题，那边改 translate.cpp。）
#
# 【补丁 2】会话中途的 system 消息
#   ZCode 会把技能清单/环境提醒作为 system 角色消息插在对话中间（实测出现在
#   messages[3]），而模型模板要求 system 必须在首条，否则抛异常：
#     Jinja Exception: System message must be at the beginning.
#   → 表现为 53 ms 秒拒、整轮 400（与图片、长度、缓存都无关）。
#   处置：把非首条的 system 消息按 user 轮渲染（内容保留，模型仍能看到），不再抛异常。
#
# 用法:
#   ./apply-kvmem-compat.sh <某个 .gguf> [输出 .jinja]
#   默认输出: patches/bonsai-chat-template-alias.jinja（start-kvmem.sh 会自动挂上）
# =============================================================================
set -euo pipefail

GGUF="${1:?用法: $0 <model.gguf> [out.jinja]}"
OUT="${2:-$(cd "$(dirname "$0")" && pwd)/bonsai-chat-template-alias.jinja}"

python3 - "$GGUF" "$OUT" <<'PY'
import struct, sys

src, out = sys.argv[1], sys.argv[2]
f = open(src, 'rb')
magic = f.read(4)
assert magic == b'GGUF', f'不是 GGUF: {magic!r}'
struct.unpack('<I', f.read(4))          # version
struct.unpack('<Q', f.read(8))          # tensor count
n_kv, = struct.unpack('<Q', f.read(8))

def rstr():
    n, = struct.unpack('<Q', f.read(8))
    return f.read(n).decode('utf-8', 'replace')

def skip(t):
    if t in (0, 1, 7): f.read(1)
    elif t in (2, 3): f.read(2)
    elif t in (4, 5, 6): f.read(4)
    elif t in (10, 11, 12): f.read(8)
    elif t == 8: rstr()
    elif t == 9:
        et, = struct.unpack('<I', f.read(4))
        n, = struct.unpack('<Q', f.read(8))
        for _ in range(n): skip(et)
    else:
        raise SystemExit(f'未知 GGUF 值类型 {t}')

tmpl = None
for _ in range(n_kv):
    key = rstr()
    t, = struct.unpack('<I', f.read(4))
    if key == 'tokenizer.chat_template':
        tmpl = rstr()
        break
    skip(t)
if tmpl is None:
    raise SystemExit('该 GGUF 没有 tokenizer.chat_template')

report = []

# ---- 补丁 1：reasoning_effort 别名 ----
anchor1 = "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
patch1 = anchor1 + """
    {#- sm86 兼容补丁：客户端常见 high/max/minimal 等写法（ZCode 发 high）-> 归一 #}
    {%- if resolved_reasoning_effort in ('high', 'max', 'maximum', 'highest') %}{%- set resolved_reasoning_effort = 'xhigh' %}
    {%- elif resolved_reasoning_effort in ('minimal', 'min', 'lowest') %}{%- set resolved_reasoning_effort = 'low' %}
    {%- elif resolved_reasoning_effort in ('mid', 'default', 'auto') %}{%- set resolved_reasoning_effort = 'medium' %}
    {%- elif resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}{%- set resolved_reasoning_effort = 'xhigh' %}
    {%- endif %}"""
if 'resolved_reasoning_effort in (\'high\'' in tmpl:
    report.append('补丁1 已存在（跳过）')
elif anchor1 in tmpl:
    tmpl = tmpl.replace(anchor1, patch1, 1)
    report.append('补丁1 已应用（reasoning_effort 别名）')
else:
    raise SystemExit('模板结构与预期不符（找不到 effort 锚点），请人工确认后再打')

# ---- 补丁 2：会话中途的 system 消息 -> 按 user 轮渲染 ----
anchor2 = """    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}"""
patch2 = """    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {#- sm86 兼容补丁：会话中途的 system 消息（ZCode 的技能清单提醒等）按 user 轮渲染，不再抛异常 #}
            {{- '<|im_start|>user\\n' + content + '<|im_end|>\\n' }}
        {%- endif %}"""
if "System message must be at the beginning" not in tmpl:
    report.append('补丁2 已存在（跳过）')
elif anchor2 in tmpl:
    tmpl = tmpl.replace(anchor2, patch2, 1)
    report.append('补丁2 已应用（中途 system 消息按 user 渲染）')
else:
    raise SystemExit('补丁2 锚点不匹配（模板结构变了），请人工确认')

open(out, 'w', encoding='utf-8').write(tmpl)
for r in report:
    print(' ', r)
print(f'已写出 {out}（{len(tmpl)} 字符）')
PY

echo "启动时挂上（start-kvmem.sh 会自动带 --chat-template-file）:"
echo "  ./scripts/start-kvmem.sh long"
