#!/usr/bin/env bash
# =============================================================================
# KVMem 线（路线 A）兼容补丁：reasoning_effort 别名归一
#
# 问题（本机实测，2026-10-03）:
#   模型自带聊天模板只接受 reasoning_effort ∈ {xhigh, medium, low}，其它值直接
#   raise_exception → HTTP 400。而 OpenAI 系客户端默认发 "high"——
#   ZCode 就发 {"reasoning_effort":"high","enable_thinking":true}，
#   于是整轮对话被 400 拒掉：
#     Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh...
#
# 修法（不用重编引擎）:
#   从 GGUF 抽出 tokenizer.chat_template，插入一段别名归一逻辑，存成 .jinja，
#   启动时挂 --chat-template-file <该文件>。
#   映射: high/max/maximum/highest → xhigh；minimal/min/lowest → low；其余未知 → xhigh。
#   （与 ninfer 线的 patches/apply-compat.sh P1 是同一类修复：那边改 translate.cpp，
#     这边只换模板。）
#
# 用法:
#   ./apply-kvmem-compat.sh <某个 .gguf> [输出 .jinja]
#   默认输出: patches/bonsai-chat-template-alias.jinja
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

anchor = "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
if anchor not in tmpl:
    raise SystemExit('模板结构与预期不符（找不到 effort 锚点），请人工确认后再打')
if 'sm86 兼容补丁' in tmpl:
    print('模板已含别名补丁，直接写出')
    patched = tmpl
else:
    patch = anchor + """
    {#- sm86 兼容补丁：客户端常见 high/max/minimal 等写法（ZCode 发 high）-> 归一 #}
    {%- if resolved_reasoning_effort in ('high', 'max', 'maximum', 'highest') %}{%- set resolved_reasoning_effort = 'xhigh' %}
    {%- elif resolved_reasoning_effort in ('minimal', 'min', 'lowest') %}{%- set resolved_reasoning_effort = 'low' %}
    {%- elif resolved_reasoning_effort in ('mid', 'default', 'auto') %}{%- set resolved_reasoning_effort = 'medium' %}
    {%- elif resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}{%- set resolved_reasoning_effort = 'xhigh' %}
    {%- endif %}"""
    patched = tmpl.replace(anchor, patch, 1)

open(out, 'w', encoding='utf-8').write(patched)
print(f'已写出 {out}（{len(patched)} 字符）')
PY

echo "启动时挂上: KVMEM_MTP=1 ./scripts/start-kvmem.sh work   # start-kvmem.sh 会自动带 --chat-template-file"
