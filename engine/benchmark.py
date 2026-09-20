import time
import torch

from engine import Engine


MODEL_PATH = "/oscar/scratch/szeng26/.cache/huggingface/hub/Qwen3-4B-Instruct-2507"

BATCH_SIZE = 8
INPUT_TOKENS = 128
NEW_TOKENS = 256
WARMUP_RUNS = 2
BENCH_RUNS = 5


def make_inputs(batch_size, input_tokens, tokenizer):
    text = "Explain how a transformer language model works."
    ids = tokenizer.encode(text)

    # Repeat/truncate to get a controlled input length.
    ids = (ids * ((input_tokens + len(ids) - 1) // len(ids)))[:input_tokens]

    return [ids[:] for _ in range(batch_size)]


def run_generation(engine, input_ids, max_new_tokens):
    # Consume the generator completely.
    for _ in engine.generate(input_ids, max_new_tokens):
        pass


def main():
    engine = Engine(MODEL_PATH)

    # If you have a tokenizer available, use it to create realistic token IDs.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    input_ids = make_inputs(BATCH_SIZE, INPUT_TOKENS, tokenizer)

    print("Warming up...")
    for _ in range(WARMUP_RUNS):
        run_generation(engine, input_ids, NEW_TOKENS)

    torch.cuda.synchronize()

    times = []

    print("Benchmarking...")
    for i in range(BENCH_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()

        run_generation(engine, input_ids, NEW_TOKENS)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        tokens = BATCH_SIZE * NEW_TOKENS
        tok_per_sec = tokens / elapsed

        print(
            f"run {i + 1}: "
            f"{elapsed:.4f}s, "
            f"{tok_per_sec:.2f} tok/s "
            f"({tok_per_sec / BATCH_SIZE:.2f} tok/s/sequence)"
        )

    avg_time = sum(times) / len(times)
    total_tokens = BATCH_SIZE * NEW_TOKENS

    print()
    print("=== Results ===")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Input tokens:     {INPUT_TOKENS}")
    print(f"New tokens:       {NEW_TOKENS}")
    print(f"Average latency:  {avg_time:.4f}s")
    print(f"Throughput:       {total_tokens / avg_time:.2f} tok/s")
    print(f"Per sequence:     {total_tokens / avg_time / BATCH_SIZE:.2f} tok/s")


if __name__ == "__main__":
    main()
