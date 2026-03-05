# ~25M parameter byte-level BPE GPT for pre-Copernican corpus

config = {
    # data
    "block_size": 512,
    "batch_size": 32,
    "eval_batch_size": 32,
    "eval_interval": 200,
    "eval_iters": 20,
    "data_dir": "data/nanogpt/pre_copernican_bpe",
    "out_dir": "out/pre_copernican_bpe_25m",

    # model
    "n_layer": 8,
    "n_head": 8,
    "n_embd": 512,
    "dropout": 0.1,

    # optimization
    "learning_rate": 3e-4,
    "max_iters": 8000,
    "weight_decay": 0.03,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # system
    "device": "cuda"  # auto-fallback to cpu if unavailable
}

# Rough VRAM guidance (single precision):
# Parameters + Adam states ~ 300-400 MB.
# Activations scale with batch * block_size * n_embd.
