# Fine-Tuning Explanation: LoRA, QLoRA, Non-Instruction FT, SFT, and DPO

## Why full fine-tuning is expensive

Full fine-tuning updates every parameter in the model. Even a "small" 1.1B-parameter model
like TinyLlama has over a billion weights, and training requires storing not just the weights
but also gradients and optimizer states (Adam keeps two extra moving-average tensors per
parameter) for all of them. That's roughly 4x the model's own memory footprint before you've
even loaded a single training batch. For bigger models (7B, 13B, 70B+), this quickly exceeds
what a single consumer or even a single datacenter GPU can hold, and requires multi-GPU
sharding, which is expensive in both hardware and engineering time. It also means every
fine-tuning run produces a brand new full-size copy of the model to store.

## What LoRA does

LoRA (Low-Rank Adaptation) freezes all of the original model's weights and instead injects a
small pair of trainable matrices (A and B) alongside each targeted weight matrix (in this
project: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Instead
of learning a full-size weight update, the model learns a low-rank approximation of that
update: `W_new = W_frozen + (B @ A) * scaling`. Because A and B are much smaller than the
original weight matrix (rank `r=16` in this project vs. the full hidden dimension), the number
of trainable parameters drops by roughly 99% (our runs show ~12.6M trainable params out of
~1.1B total, about 1.13%). This means far less GPU memory for gradients/optimizer state, and a
much smaller file to save at the end (a few MB adapter instead of gigabytes).

## What QLoRA does

QLoRA adds one more idea on top of LoRA: it loads the frozen base model in 4-bit quantized
precision (via bitsandbytes) instead of fp16/fp32, while keeping the small LoRA adapter
matrices in higher precision for stable training. The frozen weights consume ~4x less memory
than fp16 and ~8x less than fp32, and since they're frozen anyway (no gradients needed for
them), the precision loss from quantization mostly doesn't hurt training quality.

## Why QLoRA is useful on limited GPU

On a memory-constrained GPU (e.g., a free-tier Colab T4 with ~15GB VRAM), QLoRA lets you fit
and fine-tune much larger models than would otherwise fit, because the bulk of the model's
memory (the frozen weights) is quantized down to a quarter or eighth of its normal size. For
this project specifically, we actually moved *away* from QLoRA's 4-bit loading for TinyLlama-1.1B,
because the model is small enough that plain LoRA on the full-precision model never exceeded
~5GB of the T4's 14.5GB VRAM budget - the 4-bit+fp16 combination was actually causing NaN
gradients (documented in `discussions.md`), not saving us anything we needed. QLoRA's value is
proportional to how memory-constrained you are relative to model size; for a small model on a
T4, plain LoRA was both simpler and more numerically stable.

## What is non-instruction fine-tuning?

Non-instruction fine-tuning (Stage 1) trains the model on raw domain text with no
question/answer structure - just plain paragraphs, using a standard causal language modeling
objective (predict the next token). There's no prompt/response format; the model simply learns
to better predict domain-specific language, phrasing, and recurring facts (e.g., "Monday to
Friday, 9 AM to 6 PM" for support hours) by being exposed to more of it. This is analogous to
how base LLMs are pretrained, just on a small, focused domain corpus.

## What is instruction fine-tuning?

Instruction fine-tuning (Stage 2, also called SFT - Supervised Fine-Tuning) trains the model
on structured `instruction -> response` pairs, formatted with a template (`### Instruction:
... ### Response: ...` in this project). This teaches the model the *behavior* of answering a
question directly and helpfully, rather than just continuing text. It builds on top of Stage
1's domain-adapted weights, so the model already "knows" the domain and now additionally learns
the question-answering format and tone expected of the assistant.

## What is DPO?

DPO (Direct Preference Optimization) is a preference-alignment technique that trains the model
using pairs of responses to the same prompt: a `chosen` (better) response and a `rejected`
(worse) response. Instead of needing a separate reward model (like classic RLHF), DPO directly
optimizes the policy model so the log-probability margin between the chosen and rejected
response grows - i.e., the model becomes more likely to produce the chosen-style answer and
less likely to produce the rejected-style one, relative to a reference (frozen) copy of the
same model. In this project, `ref_model=None` was used, which lets the DPO trainer treat the
un-trained LoRA adapter's initial state as the implicit reference (a common shortcut for
PEFT-based DPO, since disabling the adapter recovers the pre-DPO model as the reference).

## Difference between SFT and DPO

SFT teaches the model *what* a good response looks like by showing it examples of good
responses directly (positive-only supervision). DPO teaches the model a *relative preference*
- given two candidate responses, prefer this one over that one - without ever specifying the
single "correct" response. SFT is about learning the target behavior/format from scratch; DPO
is about refining and sharpening an already-capable model's judgment between a better and a
worse answer, which is why it's applied as the third stage, after the model already knows how
to answer questions from SFT.

## Hyperparameters used

All three stages share the same LoRA target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, `down_proj`), rank, alpha, and dropout; only the learning rate, step
count, and (for Stage 3) the DPO-specific beta differ:

| Parameter | Stage 1 (Non-Instruction) | Stage 2 (Instruction/SFT) | Stage 3 (DPO) |
|---|---|---|---|
| Base model | `unsloth/tinyllama-bnb-4bit` (TinyLlama-1.1B) | same | same |
| LoRA rank (`r`) | 16 | 16 | 16 |
| LoRA alpha | 32 | 32 | 32 |
| LoRA dropout | 0 | 0 | 0 |
| Target modules | q/k/v/o_proj, gate/up/down_proj | same | same |
| Learning rate | 3e-4 | 1.5e-4 | 5e-5 |
| Batch size (per device) | 1 | 1 | 1 |
| Gradient accumulation steps | 8 (effective batch = 8) | 8 | 8 |
| Max steps | 100 | 150 | 120 |
| Warmup steps | 10 | 10 | 10 |
| Optimizer | adamw_torch | adamw_torch | adamw_torch |
| DPO beta | - | - | 0.1 |
| Precision | fp32, no quantization | fp32, no quantization | fp16, no quantization* |
| Trainable params | ~12.6M / ~1.1B (1.13%) | ~12.6M / ~1.1B (1.13%) | ~12.6M / ~1.1B (1.13%) |

\* Stage 3 required loading the base model in fp16 rather than fp32 - `DPOTrainer` forces an
active fp16 autocast context internally on this T4 GPU regardless of config, and Unsloth's
fused LoRA kernel reads raw weight tensors directly (bypassing autocast's automatic per-op
casting), so the base model's own dtype has to match the forced autocast dtype to avoid a
`Half`/`Float` mismatch. Full details of this investigation are in `discussions.md`.

Learning rate was deliberately decreased at each stage (3e-4 -> 1.5e-4 -> 5e-5): non-instruction
fine-tuning needs the largest push since the model is adapting to an entirely new domain
vocabulary from scratch, instruction tuning needs a gentler nudge since it's teaching format on
top of already-adapted knowledge, and DPO alignment needs the gentlest touch of all since it's
fine-grained preference shaping on top of an already-competent model.
