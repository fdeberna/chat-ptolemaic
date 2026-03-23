# ~25M parameter character-level GPT for pre-Copernican corpus
# Est. params: ~25M (n_layer=8, n_head=8, n_embd=512, vocab ~150)
# Block size 512 for longer context.

config = {
    # data
    "block_size": 512,
    "batch_size": 32,
    "eval_batch_size": 32,
    "eval_interval": 200,
    "eval_iters": 20,
    "data_dir": "data/nanogpt/pre_copernican",
    "out_dir": "out/pre_copernican_25m",

    # model
    "n_layer": 8,
    "n_head": 8,
    "n_embd": 512,
    "dropout": 0.15,

    # optimization
    "learning_rate": 3e-4,
    "max_iters": 20000,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # system
    "device": "cuda"  # auto-fallback to cpu if unavailable
}

# Rough VRAM guidance (single precision):
# - Parameters + Adam states ~ 300-400 MB.
# - Activations scale with batch * block_size * n_embd.
#   For batch=32, block=512: ~32*512*512 floats ≈ 8.4M -> ~34 MB forward.
#   With attention/MLP buffers and backward passes, expect ~6-8x -> ~200-300 MB.
#   Total footprint typically fits in ~1-2 GB; allow extra headroom for PyTorch/cuda.
