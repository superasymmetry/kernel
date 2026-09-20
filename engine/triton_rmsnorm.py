"""Fused RMSNorm forward in Triton, drop-in for Qwen3RMSNorm.

Matches the HF reference numerically: accumulate in fp32, round the
normalized value to the input dtype, then scale by the weight.

Registered as a torch custom op so `torch.compile` can trace through it
without a graph break (a graph break would disable CUDA graphs, which
cost far more than this kernel saves).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd(
    X, Y, W,
    stride_x, stride_y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd).to(Y.dtype.element_ty).to(tl.float32) * w

    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@torch.library.custom_op("engine::rmsnorm", mutates_args=())
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    shape = x.shape
    x2d = x.reshape(-1, shape[-1])
    x2d = x2d if x2d.is_contiguous() else x2d.contiguous()
    N = x2d.shape[-1]
    y = torch.empty_like(x2d)

    BLOCK = triton.next_power_of_2(N)
    num_warps = max(4, min(16, BLOCK // 256))
    _rmsnorm_fwd[(x2d.shape[0],)](
        x2d, y, weight,
        x2d.stride(0), y.stride(0),
        N, eps,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return y.reshape(shape)


@rmsnorm.register_fake
def _(x, weight, eps):
    return torch.empty_like(x)


def patch_qwen3_rmsnorm() -> None:
    """Replace Qwen3RMSNorm.forward with the fused kernel.

    Only worth it when running eager. Under `torch.compile` the custom op is
    opaque, so inductor cannot fuse the norm into the neighbouring residual
    add / matmul epilogue the way it can with the reference implementation --
    leave this off when compiling.
    """
    from transformers.models.qwen3 import modeling_qwen3

    def forward(self, hidden_states):
        if hidden_states.device.type != "cuda":
            return _reference(self, hidden_states)
        return rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    def _reference(self, hidden_states):
        dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * h.to(dtype)

    modeling_qwen3.Qwen3RMSNorm.forward = forward
