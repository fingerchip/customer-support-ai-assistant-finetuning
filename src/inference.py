"""
Simple inference script for the Customer Support AI Assistant.

Loads the base model (unsloth/tinyllama-bnb-4bit), reconstructs the trained
chain by merging the Stage 1 (non-instruction) and Stage 2 (instruction/SFT)
adapters, then attaches the Stage 3 (DPO) adapter on top - this is the exact
loading pattern used and verified in notebooks/dpo_alignment.ipynb's
fresh-reload sanity check.

Usage:
    python src/inference.py "How can I cancel my order?"

    # or interactively:
    python src/inference.py
    > Ask a question (blank line to quit): How can I track my order?
"""

import os
import sys

import torch
from peft import PeftModel
from unsloth import FastLanguageModel

BASE_MODEL_NAME = "unsloth/tinyllama-bnb-4bit"
MAX_SEQ_LENGTH = 512

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE1_ADAPTER_DIR = os.path.join(REPO_ROOT, "models", "non_instruction_adapter")
STAGE2_ADAPTER_DIR = os.path.join(REPO_ROOT, "models", "instruction_adapter")
STAGE3_ADAPTER_DIR = os.path.join(REPO_ROOT, "models", "dpo_adapter")

# Loaded in fp16 (not fp32): this matches the dtype the Stage 3 adapter was
# actually trained under (see discussions.md items 24-25 for why fp16 was
# required there), so reconstructing the chain in fp16 for inference keeps
# the merge numerically consistent with training.
_MODEL_DTYPE = torch.float16


def build_instruction_prompt(instruction: str, input_text: str = "") -> str:
    instruction = str(instruction).strip()
    input_text = str(input_text).strip()

    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def load_final_assistant():
    """Reconstructs base -> Stage 1 merged -> Stage 2 merged -> Stage 3 (DPO) adapter."""
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=_MODEL_DTYPE,
        load_in_4bit=False,
    )

    stage1_model = PeftModel.from_pretrained(base_model, STAGE1_ADAPTER_DIR)
    stage1_merged = stage1_model.merge_and_unload()

    stage2_model = PeftModel.from_pretrained(stage1_merged, STAGE2_ADAPTER_DIR)
    stage2_merged = stage2_model.merge_and_unload()

    final_model = PeftModel.from_pretrained(stage2_merged, STAGE3_ADAPTER_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(final_model)
    return final_model, tokenizer


_model = None
_tokenizer = None


def generate_answer(question: str, max_new_tokens: int = 150) -> str:
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load_final_assistant()

    prompt = build_instruction_prompt(question)
    inputs = _tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    input_tokens = inputs["input_ids"].shape[-1]
    generated_tokens = output[0][input_tokens:]
    return _tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer = generate_answer(question)
        print(answer)
    else:
        while True:
            question = input("Ask a question (blank line to quit): ").strip()
            if not question:
                break
            answer = generate_answer(question)
            print(answer)
            print("---")
