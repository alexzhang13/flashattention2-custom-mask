import pytest
import torch
import torch.nn.functional as F
import triton

from fa2_custom_mask import flash_attention_custom_mask

@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 128, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op_causal(Z, H, N_CTX, causal, HEAD_DIM, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    if causal:
        mask = torch.tril(torch.ones((Z, H, N_CTX, N_CTX), dtype=torch.uint8, device="cuda", requires_grad=False))
    else:
        mask = None
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = flash_attention_custom_mask(q, k, v, mask, sm_scale).half()
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of MI200 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 1e-2
    
    assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
    assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
    assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)

    
@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 128, 64)])
def test_op_random(Z, H, N_CTX, HEAD_DIM, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    
    # create random mask
    M = mask = torch.randint(0, 2, (N_CTX, N_CTX), dtype=torch.uint8, device="cuda", requires_grad=False)
    mask = torch.broadcast_to(mask, (Z, H, N_CTX, N_CTX))
    p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = flash_attention_custom_mask(q, k, v, mask, sm_scale).half()
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of MI200 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 1e-2
    
    assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
    assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
    assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False
    
try:
    from fa2_original import attention
    USE_FA2_TRITON_ORIGINAL = True
except BaseException:
    USE_FA2_TRITON_ORIGINAL = False
    
try:
    import xformers
    import xformers.ops
    from xformers.ops import memory_efficient_attention
    import xformers.ops.fmha as fmha
    HAS_XFORMERS = True
except BaseException:
    HAS_XFORMERS = False 


TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, HEAD_DIM = 4, 16, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True, False]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(8, 15)],
                line_arg="provider",
                line_vals=["triton_custom_mask-fp16"] + (["triton_custom_mask-fp8"] if TORCH_HAS_FP8 else []) +
                (["flash"] if HAS_FLASH else []) + (["triton-original-fp16"] if USE_FA2_TRITON_ORIGINAL else []) + (["xformers-memory_efficient_attention"] if HAS_XFORMERS else []),
                line_names=["Triton (Custom Mask) [FP16]"] + (["Triton (Custom Mask) [FP8]"] if TORCH_HAS_FP8 else []) +
                (["Flash-2"] if HAS_FLASH else []) + (["Original Triton [FP16]"] if USE_FA2_TRITON_ORIGINAL else []) + (["XFormers Memory-Efficient Attn"] if HAS_XFORMERS else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-"), ("pink", "-")],
                ylabel="GFLOPS",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    dtype = torch.float16
    if "triton_custom_mask" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if causal:
            mask = torch.tril(torch.ones((BATCH, H, N_CTX, N_CTX), dtype=torch.uint8, device=device, requires_grad=False))
        else:
            mask = None
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: flash_attention_custom_mask(q, k, v, mask, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    elif "triton-original" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "xformers-memory_efficient_attention":
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            fn = lambda: F.scaled_dot_product_attention(q, k, v)
            # fn = lambda: flash_attn_func(qkv, causal=causal)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal and provider in ["flash", "triton-original-fp16"]:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9

configs_random = []
for mode in ["fwd", "bwd"]:
    configs_random.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[2**i for i in range(8, 15)],
            line_arg="provider",
            line_vals=["triton_custom_mask-fp16"] + (["triton_custom_mask-fp8"] if TORCH_HAS_FP8 else []) + (["xformers-memory_efficient_attention"] if HAS_XFORMERS else []),
            line_names=["Triton (Custom Mask) [FP16]"] + (["Triton (Custom Mask) [FP8]"] if TORCH_HAS_FP8 else []) + (["XFormers Memory-Efficient Attn"] if HAS_XFORMERS else []),
            styles=[("red", "-"), ("blue", "-"), ("pink", "-")],
            ylabel="GFLOPS",
            plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-random-mask",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
                "mode": mode,
            },
        ))

@triton.testing.perf_report(configs_random)
def bench_flash_attention_random_mask(BATCH, H, N_CTX, HEAD_DIM, mode, provider, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    dtype = torch.float16
    if "triton_custom_mask" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        mask = torch.randint(0, 2, (BATCH, 1, N_CTX, N_CTX), dtype=dtype, device="cuda", requires_grad=False)
        mask = torch.broadcast_to(mask, (BATCH, H, N_CTX, N_CTX))
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: flash_attention_custom_mask(q, k, v, mask, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "xformers-memory_efficient_attention":
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        mask = torch.randint(0, 2, (BATCH, 1, N_CTX, N_CTX), dtype=dtype, device="cuda", requires_grad=False) # doesn't allow uint8
        mask = torch.broadcast_to(mask, (BATCH, H, N_CTX, N_CTX))
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
        fn = lambda: fmha.memory_efficient_attention(q, k, v, attn_bias=mask)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal and provider in ["flash", "triton-original-fp16"]:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention_random_mask.run(save_path="data/", print_data=True)
    bench_flash_attention.run(save_path="data/", print_data=True)
