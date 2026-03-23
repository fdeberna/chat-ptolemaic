# Byte-level BPE GPT for pre-Copernican corpus

config = {
    # data
    "block_size": 256,          # context length in tokens
    "batch_size": 64,
    "eval_batch_size": 64,
    "eval_interval": 200,
    "eval_iters": 20,
    "data_dir": "data/nanogpt/pre_copernican_bpe",
    "out_dir": "out/pre_copernican_bpe",

    # model
    "n_layer": 8,
    "n_head": 8,
    "n_embd": 512,
    "dropout": 0.2,

    # optimization
    "learning_rate": 3e-4,
    "lr_scheduler_type": "cosine",
    "early_stop_patience": 3,
    "max_iters": 5000,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # system
    "device": "cuda"  # will auto-fallback to cpu if not available
}


# # Corpus
# Total: 8M tokens (after cleaning 10% noise)
# Shuffle split: 90/10
#
# # Model
# n_layer = 8
# n_head = 8
# n_embd = 448
# dropout = 0.3  # High!
#
# # ~20M params
# # Ratio: 8M / 20M = 0.40 tokens/param
