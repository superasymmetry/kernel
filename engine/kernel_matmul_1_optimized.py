"""
AutoKernel -- Extracted kernel from model profiling.
Op type: matmul
Rank: 1 (8.2% of GPU time)
Model shape: M=8, N=9728, K=2560 (gate_proj / up_proj, batch-8 decode)

This kernel was extracted from profiling models/qwen3_4b.py.
The agent optimizes this to maximize throughput at the model-specific shapes.

OPTIMIZED (Phase B, experiments 1-10): 279.96 -> 568.04 TFLOPS on the primary
bench size (2048^3 fp16), 0.47x -> 0.97x of cuBLAS, 37% -> 75% of fp16 peak.
Key changes: fp32 uses IEEE dot precision, 128x256x64 tile with num_stages=4,
grouped L2 swizzle, and a 64x64 fallback when the big tile cannot fill a wave.
"""

KERNEL_TYPE = "matmul"

# Model-specific shapes (the shapes that matter for THIS model)
MODEL_SHAPES = {'M': 8, 'N': 9728, 'K': 2560}  # Qwen3-4B decode, batch 8: gate_proj / up_proj

# Benchmark config (self-describing -- bench.py can load this dynamically)
TEST_SIZES = [
    ("model_primary", {'M': 2048, 'N': 2048, 'K': 2048}),
    # Also test nearby sizes for robustness
    ("model_half", {'M': 1024, 'N': 1024, 'K': 1024}),
    ("model_double", {'M': 4096, 'N': 4096, 'K': 4096}),
]

TOLERANCES = {'float16': {'atol': 0.01, 'rtol': 0.01}, 'bfloat16': {'atol': 0.02, 'rtol': 0.02}, 'float32': {'atol': 0.0001, 'rtol': 0.0001}}


def FLOPS_FN(s):
    return 2 * s["M"] * s["N"] * s["K"]


def BYTES_FN(s, dt_bytes):
    return (s["M"] * s["K"] + s["K"] * s["N"] + s["M"] * s["N"]) * dt_bytes


# ======================================================================
# Triton kernel code (from kernels/matmul.py)
# ======================================================================

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PRECISION: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Basic tiled matmul. The agent improves this."""
    # Grouped tile ordering: walk GROUP_SIZE_M rows of tiles before advancing in
    # N, so concurrent blocks reuse the same A/B panels out of L2.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision=PRECISION)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k += BLOCK_SIZE_K

    c = acc.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


_SM_COUNT = None


def _sm_count(device) -> int:
    global _SM_COUNT
    if _SM_COUNT is None:
        _SM_COUNT = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNT


def kernel_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.matmul_ref signature."""
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # fp32 inputs must use true IEEE fp32 math: Triton's default tf32 dot loses
    # ~10 mantissa bits, which blows past the 1e-4 fp32 tolerance.
    is_fp32 = A.dtype == torch.float32
    PRECISION = "ieee" if is_fp32 else "tf32"

    # A 128x128 tile only pays off when it produces enough tiles to fill the GPU.
    # At M=N=1024 it yields 64 blocks on 132 SMs -- half the machine idle -- so
    # drop to 64x64 whenever the big tile cannot cover one full wave.
    if triton.cdiv(M, 128) * triton.cdiv(N, 128) < _sm_count(A.device):
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        num_warps = 4
        num_stages = 4
    else:
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 256
        num_warps = 8
        num_stages = 4

    # fp32 tiles are 2x the bytes, so halve K to stay inside 228 KB of smem.
    BLOCK_SIZE_K = 32 if is_fp32 else 64

    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        PRECISION=PRECISION,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C
