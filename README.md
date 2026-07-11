# Customer Support AI Assistant - Fine-Tuning with Unsloth

## 1. Project Title

**Customer Support AI Assistant** - a domain-specific LLM assistant built by fine-tuning
TinyLlama-1.1B through a 3-stage pipeline (non-instruction fine-tuning -> instruction fine-tuning
-> DPO preference alignment) using [Unsloth](https://github.com/unslothai/unsloth).

## 2. Domain Selected

**Customer Support Assistant.**

## 3. Business Problem

As a GenAI Engineer, the task is to build an internal AI assistant that can answer customer
support questions clearly and specifically for one company - covering refunds, order tracking,
product issues, cancellations, replacements, and payment issues - giving noticeably better,
more specific answers than an untuned base model would.

## 4. Dataset Details

All three datasets are derived from the public
[Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(Hugging Face), cleaned to replace its `{{Placeholder}}` template tokens with concrete
values (e.g., `{{Customer Support Hours}}` -> "Monday to Friday, 9 AM to 6 PM").

| File | Rows | Source field(s) | Purpose |
|---|---|---|---|
| `data/non_instruction_data.txt` | 70 paragraphs | `response` | Stage 1 raw domain text |
| `data/instruction_dataset.jsonl` | 120 pairs | `instruction` + `response` | Stage 2 instruction/response pairs |
| `data/preference_dataset.jsonl` | 60 triples | `instruction`, `response` (chosen) + generic templates (rejected) | Stage 3 DPO preference triples |

For the preference dataset, `chosen` responses are the real, cleaned Bitext answers; `rejected`
responses are drawn from a fixed pool of 10 deliberately generic/dismissive templates (e.g.
"That's not something we handle here.") - clearly worse without being unsafe, per the
assignment's guidance.

## 5. Base Model Used

`unsloth/tinyllama-bnb-4bit` (TinyLlama-1.1B), loaded via Unsloth's `FastLanguageModel`.
Despite the `-bnb-4bit` name (Unsloth's naming convention for its pre-quantized repos), all
three stages load it **without** 4-bit quantization (`load_in_4bit=False`) - see Section 9 and
`discussions.md` for why.

## 6. Non-Instruction Fine-Tuning Approach (Stage 1)

Trains the model with a plain causal-language-modeling objective directly on the 70 cleaned
domain paragraphs (no prompt/response structure) so it absorbs domain vocabulary, recurring
facts, and phrasing before ever seeing instruction-formatted data. Uses a fresh LoRA adapter
on top of the base model, trained with vanilla HF `Trainer` + `DataCollatorForLanguageModeling`.

**Notebook:** `notebooks/non_instruction_finetuning.ipynb`

## 7. Instruction Fine-Tuning Approach (Stage 2)

Reconstructs Stage 1's trained model (base + Stage 1 adapter, merged), applies a **new** LoRA
adapter on top, and trains on the 120 `instruction -> response` pairs formatted as
`### Instruction: ... ### Response: ...`. This teaches the model the question-answering
behavior/format on top of the domain knowledge already absorbed in Stage 1.

**Notebook:** `notebooks/instruction_finetuning.ipynb`

## 8. DPO Alignment Approach (Stage 3)

Reconstructs the full chain (base -> Stage 1 adapter merged -> Stage 2 adapter merged), applies
a fresh LoRA adapter, and runs `DPOTrainer` (TRL) on the 60 `(prompt, chosen, rejected)`
triples with `beta=0.1` and `ref_model=None` (the un-trained adapter state serves as the
implicit reference). This sharpens the model's preference for the professional, specific
`chosen` answers over the generic `rejected` ones.

**Notebook:** `notebooks/dpo_alignment.ipynb`

## 9. LoRA / QLoRA Configuration

All three stages share the same LoRA target modules, rank, alpha, and dropout:

| Parameter | Stage 1 | Stage 2 | Stage 3 (DPO) |
|---|---|---|---|
| LoRA rank (`r`) | 16 | 16 | 16 |
| LoRA alpha | 32 | 32 | 32 |
| LoRA dropout | 0 | 0 | 0 |
| Target modules | q/k/v/o_proj, gate/up/down_proj | same | same |
| Learning rate | 3e-4 | 1.5e-4 | 5e-5 |
| Effective batch size | 8 (1 x 8 grad-accum) | 8 | 8 |
| Max steps | 100 | 150 | 120 |
| DPO beta | - | - | 0.1 |
| Precision | fp32, no quantization | fp32, no quantization | fp16, no quantization |
| Trainable params | ~12.6M / ~1.1B (1.13%) | same | same |

QLoRA (4-bit quantization) was deliberately **not** used for the final recipe: TinyLlama-1.1B
is small enough that plain LoRA never exceeded ~5GB of the T4's 14.5GB VRAM, and 4-bit+fp16
was actually causing NaN gradients during early experiments (see Section 13 and
`discussions.md`). Full reasoning for every hyperparameter is in
`reports/fine_tuning_explanation.md`.

## 10. Training Logs

**Stage 1 (Non-Instruction):** gradient norm 15.98 (healthy), average loss on 5 held-out
domain paragraphs **3.0962 -> 0.0938** before/after training, confirmed identical (0.0938) on
a freshly-reloaded adapter.

**Stage 2 (Instruction/SFT):** gradient norm 5.42, average loss on 5 instruction pairs
**2.5392 -> 0.2260** before/after, confirmed identical on fresh reload.

**Stage 3 (DPO):** per-step training loss trajectory `0.6931 -> 0.6818 -> 0.632 -> ... -> ~0.0001`
across 120 steps (clean, monotonic, no NaNs); chosen-vs-rejected log-probability margin
**-147.6937 -> -60.0984** (a **+87.6** improvement), confirmed to **+87.57** on a freshly
reloaded adapter.

Full console logs (Unsloth banners, per-step loss, VRAM usage, gradient checks) are captured in
the executed notebook outputs.

## 11. Before vs After Output Comparison

**Base model (untrained), completion-style prompt** - `"To cancel your order, you should"`:
> "...get in touch with the seller and discuss it. How to know whether the order is a fake or
> not? ...The 50 Best Online Casinos in the USA 2018..." *(generic, drifts into unrelated
> topics - no domain knowledge)*

**After Stage 1 (non-instruction fine-tuning), same prompt:**
> "...reach out to our customer support team at 123 Exit. Your satisfaction is our top priority
> ... login to your account, navigate to the checkout or order history section, locate the
> purchase with the order number (order #5649), click on the 'Orders Section'..." *(now
> produces coherent, on-domain support language)*

**Before Stage 2 (instruction tuning), Q: "How can I cancel my order?"**
> `{ "error": { "message": "Order Cancellation Failed", "code": 400 } }` *(malformed,
> non-conversational)*

**After Stage 2, same question:**
> "We appreciate your inquiry ... please provide us with your order details or any specific
> steps you have already taken. We are here to help you throughout the process and ensure your
> order is cancelled promptly..." *(direct, conversational, on-topic answer)*

**Before Stage 3 (DPO), margin (chosen vs rejected) = -147.69. After Stage 3, margin = -60.10**
(+87.6 improvement) - the model now much more strongly prefers professional, specific answers
over generic/dismissive ones. See `notebooks/dpo_alignment.ipynb` for full generation examples.

See `reports/base_model_evaluation.md`, `reports/sft_model_comparison.md`, and
`reports/final_evaluation.md` for the full 10-question comparison across all three stages.

## 12. Final Observations

- Each stage measurably improved on the previous one by a real, quantitative metric (loss for
  Stages 1-2, chosen-vs-rejected margin for Stage 3) - not just "looks better" qualitatively.
- Non-instruction fine-tuning alone was enough to make completions domain-flavored, but not
  yet conversational; instruction fine-tuning added the question-answering behavior; DPO
  sharpened the preference for the better-quality style of answer.
- On a small model like TinyLlama-1.1B, plain LoRA (no quantization) was both simpler and more
  stable than QLoRA on this hardware.

## 13. Challenges Faced

- **4-bit + fp16 QLoRA produced NaN gradients** on this T4 GPU for Stage 1; switching to plain
  LoRA with the model loaded in fp32 (no quantization) fixed it.
- **`SFTTrainer` showed flat, non-decreasing loss** across several configurations; switching to
  vanilla HF `Trainer` + `DataCollatorForLanguageModeling` gave a real, decreasing loss curve.
- **Stage 3 (DPO) hit a persistent `RuntimeError: self and mat2 must have the same dtype, but
  got Half and Float`** across six consecutive debugging attempts (removing Unsloth's
  `PatchDPOTrainer`, disabling autocast, resetting Accelerate's singleton state, disk-roundtrip
  merges, explicit `gradient_checkpointing=False`). Root cause: `DPOTrainer` forces an active
  fp16 autocast context on this GPU regardless of config, and Unsloth's fused LoRA kernel reads
  raw weight tensors directly (bypassing autocast's automatic casting) - so the base model's
  own dtype has to match. Loading the Stage 3 base model in fp16 (instead of fp32) finally
  resolved it. The full blow-by-blow debugging log is in `discussions.md`.
- **Colab sessions are ephemeral and git-based handoff between 3 separate notebooks** (no
  persistent local GPU) required carefully saving only small LoRA adapters to the repo (not
  full merged models, which are too large for git) and reconstructing the full chain by
  re-merging adapters at the start of each subsequent stage.

## 14. Future Improvements

- Try a larger base model (e.g., Qwen2.5-1.5B or Llama-3.2-1B) now that the pipeline is proven.
- Expand the preference dataset with harder negative examples (plausible-but-wrong answers,
  not just generic dismissals) to sharpen DPO's signal further.
- Add an automated/LLM-judged scoring pass for the comparison tables instead of only manual
  qualitative review.
- Package the final merged model for lightweight local CPU inference (e.g., GGUF export) so
  the assistant doesn't require a GPU at inference time.

## Repository Structure

```
customer-support-ai-assistant-finetuning/
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
├── src/
│   └── inference.py
├── models/
│   ├── non_instruction_adapter/
│   ├── instruction_adapter/
│   └── dpo_adapter/
├── README.md
└── requirements.txt
```

## Running Inference

```bash
pip install -r requirements.txt
python src/inference.py "How can I cancel my order?"
```

Requires a CUDA GPU (developed and tested on a Colab T4).
