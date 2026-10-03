#!/usr/bin/env python3
"""路线 A（llama.cpp + KVMem）基准：decode / prefill / 长上下文召回。
用法: python3 bench_kvmem.py [base_url]      # 默认 http://127.0.0.1:18202
注意: 测速看长输出（≥300 token）——短样本被每轮固定开销主导，会低估 decode。
"""
import json, random, sys, time, urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18202").rstrip("/")
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))

WORDS = ("history empire river stone bridge forest signal window garden language memory "
         "pattern surface shadow flight anchor harvest lattice cipher beacon meadow current "
         "fabric harbor lantern marble orbit pillar quartz ridge summit timber umbrella velvet "
         "whisper xenon yarn zephyr anvil breeze candle dagger ember falcon granite hollow").split()


def synth(n_words, seed=42):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def chat(prompt, max_tokens=64, thinking=False, timeout=1800):
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    t0 = time.time()
    with NO_PROXY.open(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t0


def show(tag, d, wall):
    t = d.get("timings") or {}
    u = d.get("usage") or {}
    print(f"[{tag}] prompt={u.get('prompt_tokens')} tok  out={u.get('completion_tokens')} tok  "
          f"prefill={t.get('prompt_per_second', 0):.1f} t/s  decode={t.get('predicted_per_second', 0):.2f} t/s  "
          f"wall={wall:.1f}s")
    return t


print(f"== KVMem 基准 @ {BASE} ==\n")

# 1) decode：长输出
d, w = chat("请写一篇约 600 词的中文说明文，主题是「为什么分布式系统需要共识算法」。直接输出正文。",
            max_tokens=700, thinking=False)
show("decode-600", d, w)
print("   sample:", d["choices"][0]["message"]["content"][:80].replace("\n", " "), "\n")

# 2) decode：英文长输出（对照）
d, w = chat("Write about 600 words explaining how a modern GPU memory hierarchy works. Output the text only.",
            max_tokens=700, thinking=False)
show("decode-600-en", d, w)

# 3) prefill：~6k token 提示
d, w = chat(synth(6000) + "\n\nQuestion: reply with exactly the word ALPHA and nothing else.",
            max_tokens=8, thinking=False)
show("prefill-6k", d, w)

# 4) 长上下文召回（KVMem 主场）：~32k
needle = "The secret access code is ZEBRA-7741."
body = synth(24000) + "\n\n" + needle + "\n\n" + synth(8000) + \
       "\n\nQuestion: what is the secret access code? Answer with the code only."
d, w = chat(body, max_tokens=24, thinking=False)
show("recall-32k", d, w)
print("   answer:", d["choices"][0]["message"]["content"].strip()[:80], "| 期望含 ZEBRA-7741\n")

# 5) 长上下文召回：~96k（分段喂，注意 KVMem 窗口=24K）
body = synth(40000) + "\n\n" + needle + "\n\n" + synth(56000) + \
       "\n\nQuestion: what is the secret access code? Answer with the code only."
d, w = chat(body, max_tokens=24, thinking=False)
show("recall-96k", d, w)
print("   answer:", d["choices"][0]["message"]["content"].strip()[:80], "| 期望含 ZEBRA-7741")
