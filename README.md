# LLM Instruction Fine-Tuning with DeepSpeed ZeRO

I wanted to understand DeepSpeed ZeRO from the ground up — not just use it, but actually know what's happening across GPUs at each step. This repo is the result: a working fine-tuning pipeline for a 7B-parameter model, plus the notes and walkthroughs I wrote to build that understanding.

---

## What I set out to learn

The question I kept hitting was: *how do multiple GPUs actually share the work of training a model too big to fit on one GPU?* I'd read the ZeRO paper but I wanted to trace through the actual numbers — what bytes live where, what gets communicated, when.

So I did two things in parallel:
1. Worked through ZeRO Stage 1 and Stage 2 on a tiny hand-crafted transformer (260 parameters total) where I could track every matrix and every byte
2. Built a real training script for [Pythia-6.9B](https://huggingface.co/EleutherAI/pythia-6.9b-deduped) on Stanford Alpaca so I could see the theory translate to actual GPU memory and throughput numbers

---

## The core insight (before diving in)

ZeRO exploits the fact that in standard data-parallel training, every GPU holds an **identical** copy of the optimizer states (Adam's `m` and `v` moments, plus the FP32 master parameters). For a 7B model, that's ~84 GB of pure redundancy per GPU. ZeRO eliminates it:

- **ZeRO Stage 0**: Baseline — every GPU holds everything (~112 GB/GPU for a 7B model with Adam)
- **ZeRO Stage 1**: Shard the optimizer states only — each GPU owns 1/N of Adam's `m`, `v`, and master params (~38.5 GB/GPU on 8 GPUs)
- **ZeRO Stage 2**: Also shard the gradients — each GPU only keeps its gradient slice after reduce-scatter (~28 GB/GPU on 8 GPUs)
- **ZeRO Stage 3**: Also shard the parameters — model weights themselves are split across GPUs (~14 GB/GPU on 8 GPUs)

The surprising part: **ZeRO-1 has zero extra communication cost** over standard all-reduce, because a ring all-reduce is already a reduce-scatter followed by an all-gather. ZeRO-1 just inserts the local Adam step between those two phases.

---

## The concrete walkthroughs

These are the documents I wrote to lock in the understanding with real numbers:

### [ZeRO Stage 1 Walkthrough](docs/zero1_concrete_walkthrough.md)

I defined a tiny transformer — 4-dimensional hidden state, 2 attention heads, FFN expansion of 16, 8-token vocabulary — giving 260 total parameters. Then I traced a full training step:

- Every weight matrix written out explicitly (W_q, W_k, W_v, W_o, W_1, W_2, W_vocab, LayerNorm params)
- The forward pass computed token by token (LayerNorm → attention scores → head outputs → FFN → logits)
- Two GPUs each running their own micro-batch, producing separate gradients
- The reduce-scatter: showing exactly which GPU ends up with which 130 averaged gradient values
- The Adam update on one element (W_q[0,0]: `p = 0.12 → 0.119`) with all the FP32 arithmetic shown
- The all-gather bringing both GPUs back to a consistent full model
- Final byte counts: 2,600 bytes/GPU vs 4,160 bytes/GPU for standard data-parallel — a 37.5% saving

### [ZeRO Stage 2 Walkthrough](docs/zero2_concrete_walkthrough.md)

Same toy model, but now gradients are also sharded. After the backward pass, instead of each GPU keeping its full 260-element gradient buffer, the reduce-scatter means each GPU discards the gradient slice it doesn't own. This saves another layer of memory at the cost of slightly more careful bookkeeping.

---

## The training script

[scripts/train/deepspeed_train.py](scripts/train/deepspeed_train.py) fine-tunes Pythia-6.9B on Stanford Alpaca. The key design decisions:

- **Loss masking on prompt tokens**: labels are set to -100 for the instruction portion, so the model only takes gradient on its own completions
- **BF16 + Flash Attention 2** on capable hardware (A100/H100), falls back to eager attention otherwise
- **Gradient checkpointing** enabled to trade activation memory for recomputation
- **ZeRO stage is inferred from the config file** — swap `ds_zero0.json` → `ds_zero1.json` → `ds_zero2.json` and everything else stays the same
- **Metrics saved to JSON**: peak memory, step time, throughput, eval loss, and full loss history — so the comparison script has something real to work with

At startup the script prints the theoretical memory breakdown across stages for the loaded model:

```
┌─ Memory Budget (before sharding) ──────────────┐
│  Parameters (bf16):        13.78 GB             │
│  Gradients (bf16):         13.78 GB             │
│  Optimizer (fp32 m+v):     82.68 GB             │
│  ─────────────────────────────────────────────  │
│  Total per GPU (ZeRO-0):  110.24 GB             │
│  Total per GPU (ZeRO-1):   38.60 GB             │
│  Total per GPU (ZeRO-2):   24.82 GB             │
│  Total per GPU (ZeRO-3):   13.78 GB             │
└────────────────────────────────────────────────┘
```

### Running it

```bash
# ZeRO Stage 1 on 2 GPUs
deepspeed --num_gpus=2 scripts/train/deepspeed_train.py \
    --deepspeed_config configs/ds_zero1.json \
    --model_name EleutherAI/pythia-6.9b-deduped \
    --max_steps 500

# ZeRO Stage 2 — just change the config
deepspeed --num_gpus=2 scripts/train/deepspeed_train.py \
    --deepspeed_config configs/ds_zero2.json \
    --model_name EleutherAI/pythia-6.9b-deduped \
    --max_steps 500
```

---

## Comparing the stages

After running all three stages, [scripts/evaluation/compare_zero_stages.py](scripts/evaluation/compare_zero_stages.py) reads the saved JSON results and prints a side-by-side table:

```bash
python scripts/evaluation/compare_zero_stages.py output/
```

It also generates a chart (`output/zero_comparison.png`) with three panels: peak memory per stage, throughput per stage, and overlaid loss curves. The loss curves are the sanity check — all stages should converge to nearly identical loss, because ZeRO doesn't change the math, only the memory layout.

---

## Running inference on the fine-tuned model

```bash
python scripts/evaluation/run_inference.py \
    --model_path output/zero1 \
    --base_model EleutherAI/pythia-6.9b-deduped
```

---

## Notebook

[notebooks/DeepSpeed_ZeRO_MasterClass.ipynb](notebooks/DeepSpeed_ZeRO_MasterClass.ipynb) puts the concepts together interactively — good for stepping through the walkthrough with live code cells.

---

## Project layout

```
llm_instruction_finetuning/
├── scripts/
│   ├── train/
│   │   └── deepspeed_train.py       # main training script
│   └── evaluation/
│       ├── compare_zero_stages.py   # side-by-side comparison
│       └── run_inference.py         # generate completions from a saved model
├── configs/
│   ├── ds_zero0.json                # baseline (no sharding)
│   ├── ds_zero1.json                # shard optimizer states
│   └── ds_zero2.json                # shard optimizer + gradients
├── docs/
│   ├── zero1_concrete_walkthrough.md
│   └── zero2_concrete_walkthrough.md
├── notebooks/
│   └── DeepSpeed_ZeRO_MasterClass.ipynb
└── output/                          # training results and charts
```

## Dependencies

```
deepspeed
torch
transformers
datasets
matplotlib
```
