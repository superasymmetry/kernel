"""A/B the optimized Engine against the baseline and against HF greedy decode.

Reports tokens/sec both per-sequence and aggregate, because which one the
contest measures changes the whole strategy.
"""
import os, time, torch

MODEL = os.environ.get(
    "QWEN3_4B_PATH",
    "/oscar/scratch/szeng26/.cache/huggingface/hub/Qwen3-4B-Instruct-2507",
)
BATCH = int(os.environ.get("BATCH", 8))
PROMPT = int(os.environ.get("PROMPT", 128))
NEW = int(os.environ.get("NEW", 128))

torch.manual_seed(0)
prompts = [torch.randint(1000, 5000, (PROMPT,)).tolist() for _ in range(BATCH)]


def run(engine, label, warm):
    if warm:
        engine.warmup(BATCH, PROMPT + NEW)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    toks = list(engine.generate(prompts, NEW))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = len(toks)
    print(f"{label}: {n} steps in {dt*1000:.1f} ms | "
          f"{n/dt:.1f} tok/s per-seq | {n*BATCH/dt:.1f} tok/s aggregate | "
          f"{dt/n*1000:.2f} ms/step", flush=True)
    return toks


def reference():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()
    ids = torch.tensor(prompts, device="cuda")
    with torch.inference_mode():
        out = m.generate(ids, max_new_tokens=NEW, do_sample=False,
                         use_cache=True, pad_token_id=m.config.eos_token_id)
    del m
    torch.cuda.empty_cache()
    return out[:, PROMPT:].tolist()


ref = reference()

import baseline_engine
import engine as eng

b = run(baseline_engine.Engine(MODEL), "baseline ", warm=False)
torch.cuda.empty_cache()

o = run(eng.Engine(MODEL, compile=True), "optimized", warm=True)

# generate() yields one list-of-batch per step; transpose to per-sequence.
got = [[step[s] for step in o] for s in range(BATCH)]
old = [[step[s] for step in b] for s in range(BATCH)]
print("optimized == HF greedy :", got == ref)
print("baseline  == HF greedy :", old == ref)
for name, cand in (("optimized", got), ("baseline", old)):
    if cand != ref:
        for i, (g, r) in enumerate(zip(cand, ref)):
            if g != r:
                d = next(j for j, (x, y) in enumerate(zip(g, r)) if x != y)
                print(f"  {name} seq {i}: diverges at token {d}: {g[d]} vs {r[d]}")
                break
