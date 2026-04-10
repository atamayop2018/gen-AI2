import json
from pathlib import Path

NOTEBOOK_PATH = Path("/Users/alexitamayo/Documents/Udacity/gen-AI2/project/starter/gen_ai_fundamentals_project_starter.ipynb")
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        '''
# Generative AI Fundamentals Project: Teaching Step-by-Step Reasoning

This notebook fine-tunes `Qwen/Qwen2.5-3B-Instruct` with **LoRA** and **GRPO** so it learns to count the occurrences of a target letter in a word by reasoning step by step.

## Goals
- Build a strong prompting-only baseline.
- Create a synthetic dataset for the letter-counting task.
- Implement reward functions for numbering, spelling, counting, format, and final-answer correctness.
- Run a short sanity-check training job and a longer training job.
- Compare the original model against the tuned adapter and check for catastrophic forgetting.

> **Note:** the heavy model-loading and training cells require a CUDA GPU (for example, the course lab or a Linux T4/A10G environment). In this macOS workspace, the notebook will stay runnable, but the training cells will safely skip if CUDA is unavailable.
        '''
    ),
    code(
        '''
# Cell 1 - Optional setup for local / Colab use
# In the course GPU lab these packages may already be present.
# Uncomment the next line if you need to install the project dependencies.
# %pip install -q -r ../../requirements.txt
        '''
    ),
    code(
        '''
# Cell 2 - Environment check
import os
import sys
import shutil
import subprocess

print(f"Python executable: {sys.executable}")
print(f"Python version   : {sys.version.split()[0]}")
print(f"CUDA available   : {shutil.which('nvidia-smi') is not None}")

if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)
else:
    print("No NVIDIA GPU detected in this environment. Heavy training cells will be skipped here.")
        '''
    ),
    code(
        '''
# Cell 3 - Imports and common helpers
import math
import random
import re
from pprint import pprint

torch = None
try:
    import torch
except Exception as e:
    print(f"PyTorch is unavailable in this local environment: {e}")

import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset

LoraConfig = PeftModel = TaskType = get_peft_model = None
AutoModelForCausalLM = AutoTokenizer = None
MODEL_IMPORT_ERROR = None
try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    MODEL_IMPORT_ERROR = e
    print(f"Model-loading libraries are unavailable in this local environment: {e}")

try:
    from trl import GRPOConfig, GRPOTrainer
    GRPO_IMPORT_ERROR = None
except Exception as e:
    GRPOConfig = None
    GRPOTrainer = None
    GRPO_IMPORT_ERROR = e
    print(f"GRPO tooling is unavailable in this local environment: {e}")

SEED = 42
random.seed(SEED)

if hasattr(torch, "manual_seed"):
    torch.manual_seed(SEED)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 512
RUN_HEAVY_CELLS = bool(torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available() and AutoTokenizer is not None)

print(f"Base model       : {BASE_MODEL}")
print(f"Run heavy cells  : {RUN_HEAVY_CELLS}")
        '''
    ),
    code(
        '''
# Cell 4 - Load the tokenizer and configure the LoRA-wrapped model
#
# Choice summary:
# - lora_rank = 64 gives the adapter enough capacity to learn a new reasoning skill
#   without becoming too large or unstable. It is a strong default for a 3B model.
# - target_modules covers the key attention and MLP projection layers so the adapter
#   can influence both token-to-token reasoning and internal feature transformation.

LORA_RANK = 64
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

USE_UNSLOTH = False
unsloth_error = None

try:
    from unsloth import FastLanguageModel, PatchFastRL

    PatchFastRL("GRPO", FastLanguageModel)
    USE_UNSLOTH = True
except Exception as e:
    unsloth_error = e

# The tokenizer is light enough to load even outside the GPU environment.
tokenizer = None
if AutoTokenizer is not None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Tokenizer load skipped in this environment: {e}")
else:
    if MODEL_IMPORT_ERROR is not None:
        print(f"Tokenizer support is unavailable in this environment: {MODEL_IMPORT_ERROR}")

model = None
if RUN_HEAVY_CELLS and tokenizer is not None:
    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=TARGET_MODULES,
            lora_alpha=LORA_RANK,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        peft_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=TARGET_MODULES,
        )
        model = get_peft_model(model, peft_config)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
else:
    print("Tokenizer loaded successfully.")
    print("Skipping the 3B model load because CUDA is unavailable in this workspace.")
    if unsloth_error is not None:
        print(f"Unsloth note: {unsloth_error}")
        '''
    ),
    code(
        '''
# Cell 5 - A small helper for generating a response from the model

def build_chat_messages(system_prompt: str, user_prompt: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def render_chat(messages):
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    rendered = []
    for msg in messages:
        rendered.append(f"{msg['role'].upper()}: {msg['content']}")
    rendered.append("ASSISTANT:")
    return "\n".join(rendered)


def generate_single_response(messages, max_new_tokens: int = 180):
    if model is None:
        return "⚠️ Model not loaded in this environment. Run this notebook on a CUDA GPU to see generation output."

    prompt_text = render_chat(messages)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
        '''
    ),
    code(
        '''
# Cell 6 - Baseline: blank / weak system prompt
BASELINE_SYSTEM_PROMPT = "You are a helpful assistant."

sample_word = "effectiveness"
sample_letter = "e"
user_prompt = (
    f"Count how many times the letter '{sample_letter}' appears in the word '{sample_word}'. "
    "Show your reasoning."
)

baseline_messages = build_chat_messages(BASELINE_SYSTEM_PROMPT, user_prompt)
print(generate_single_response(baseline_messages))
        '''
    ),
    code(
        '''
# Cell 7 - Improved system prompt with Chain-of-Thought and one worked example
SYSTEM_PROMPT = """
You are a careful reasoning assistant that solves letter-counting tasks step by step.

Always follow this format exactly:
<reasoning>
1. First letter -> whether it matches -> running total
2. Second letter -> whether it matches -> running total
...
</reasoning>
<answer>final_count</answer>

Worked example:
Question: Count how many times the letter 'o' appears in the word 'room'.
<reasoning>
1. r -> not o -> running total = 0
2. o -> yes, it is o -> running total = 1
3. o -> yes, it is o -> running total = 2
4. m -> not o -> running total = 2
</reasoning>
<answer>2</answer>

Be precise. Spell the word one letter at a time, keep the running total accurate, and end with only the final numeric answer inside the answer tags.
""".strip()

improved_messages = build_chat_messages(SYSTEM_PROMPT, user_prompt)
print(generate_single_response(improved_messages))
        '''
    ),
    code(
        '''
# Cell 8 - Build a small synthetic dataset for the letter-counting task
ALL_WORDS = [
    "effectiveness", "banana", "committee", "bookkeeper", "mississippi",
    "parallel", "pepper", "letter", "deed", "assessment",
    "engineering", "balloon", "sleeveless", "coffee", "successful",
    "address", "greenhouse", "room", "queue", "cheese",
]


def generate_records(words):
    records = []
    for word in words:
        for letter in sorted(set(word)):
            answer = word.count(letter)
            prompt = f"Count how many times the letter '{letter}' appears in the word '{word}'."
            records.append(
                {
                    "word": word,
                    "letter": letter,
                    "answer": str(answer),
                    "user_prompt": prompt,
                }
            )
    return records

records = generate_records(ALL_WORDS)
print(f"Generated {len(records)} training records.")
pprint(records[:5])
        '''
    ),
    code(
        '''
# Cell 9 - Format the dataset with the system prompt

ds = Dataset.from_list(records)


def format_record(example):
    return {
        "prompt": build_chat_messages(SYSTEM_PROMPT, example["user_prompt"]),
        "word": example["word"],
        "letter": example["letter"],
        "answer": example["answer"],
    }

formatted_ds = ds.map(format_record)
formatted_ds[0]
        '''
    ),
    code(
        '''
# Cell 10 - Evaluate the untuned model on one sample from the dataset
sample_item = formatted_ds[0]
print(sample_item["prompt"][1]["content"])
print("-" * 80)
print(generate_single_response(sample_item["prompt"]))
        '''
    ),
    code(
        '''
# Cell 11 - Reward-function helpers used by all reward functions

def completion_to_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return completion.get("content", "")
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(item.get("content", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_STEP_RE = re.compile(r"^\s*(\d+)[\).:-]?\s*(.*)$", re.MULTILINE)


def extract_reasoning_and_answer(text: str):
    reasoning_match = _REASONING_RE.search(text)
    answer_match = _ANSWER_RE.search(text)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()
    answer = answer_match.group(1).strip() if answer_match else ""
    return reasoning, answer


def extract_steps(reasoning: str):
    return [(int(num), body.strip()) for num, body in _STEP_RE.findall(reasoning)]


def extract_step_letter(step_text: str):
    matches = re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", step_text)
    return matches[0].lower() if matches else None


def extract_step_total(step_text: str):
    numbers = re.findall(r"-?\d+", step_text)
    return int(numbers[-1]) if numbers else None
        '''
    ),
    code(
        '''
# Cell 12 - numbering_reward_func
# Reward in-order numbering (+0.5), penalize out-of-order numbering (-0.5),
# and penalize continuing beyond the word length (-1.0).

def numbering_reward_func(completions, word, **kwargs):
    rewards = []
    for completion, current_word in zip(completions, word):
        text = completion_to_text(completion)
        reasoning, _ = extract_reasoning_and_answer(text)
        steps = extract_steps(reasoning)

        reward = 0.0
        for expected_idx, (step_number, _step_text) in enumerate(steps, start=1):
            if step_number > len(current_word):
                reward -= 1.0
            elif step_number == expected_idx:
                reward += 0.5
            else:
                reward -= 0.5

        rewards.append(reward)
    return rewards


correct_sample = """<reasoning>
1. r -> not o -> running total = 0
2. o -> yes -> running total = 1
3. o -> yes -> running total = 2
4. m -> not o -> running total = 2
</reasoning>
<answer>2</answer>"""

incorrect_sample = """<reasoning>
1. r -> not o -> running total = 0
3. o -> yes -> running total = 1
2. x -> yes -> running total = 2
5. m -> not o -> running total = 2
</reasoning>
<answer>2</answer>"""

print(numbering_reward_func([correct_sample, incorrect_sample], word=["room", "room"]))
        '''
    ),
    code(
        '''
# Cell 13 - spelling_reward_func
# Reward exactly correct spelling (+2.0) and penalize spelling issues.

def spelling_reward_func(completions, word, **kwargs):
    rewards = []
    for completion, current_word in zip(completions, word):
        text = completion_to_text(completion)
        reasoning, _ = extract_reasoning_and_answer(text)
        steps = extract_steps(reasoning)

        spelled_letters = []
        for _, step_text in steps:
            letter = extract_step_letter(step_text)
            if letter is not None:
                spelled_letters.append(letter)

        target_letters = list(current_word.lower())
        reward = 0.0

        if spelled_letters == target_letters:
            reward += 2.0
        else:
            length_diff = abs(len(spelled_letters) - len(target_letters))
            reward -= 0.5 * length_diff

            extra_letters = max(0, len(spelled_letters) - len(target_letters))
            missing_letters = max(0, len(target_letters) - len(spelled_letters))
            reward -= 1.0 * extra_letters
            reward -= 0.5 * missing_letters

            for predicted, expected in zip(spelled_letters, target_letters):
                if predicted == expected:
                    reward += 0.2
                else:
                    reward -= 0.5

        rewards.append(reward)
    return rewards

print(spelling_reward_func([correct_sample, incorrect_sample], word=["room", "room"]))
        '''
    ),
    code(
        '''
# Cell 14 - counting_reward_func
# Reward accurate running totals (+1.0) and penalize inaccurate totals (-1.0).

def counting_reward_func(completions, word, letter, **kwargs):
    res = []
    for completion, current_word, target_letter in zip(completions, word, letter):
        text = completion_to_text(completion)
        reasoning, _ = extract_reasoning_and_answer(text)
        steps = extract_steps(reasoning)

        reward = 0.0
        running_total = 0

        for idx, (_, step_text) in enumerate(steps[: len(current_word)]):
            if current_word[idx].lower() == target_letter.lower():
                running_total += 1

            predicted_total = extract_step_total(step_text)
            if predicted_total == running_total:
                reward += 1.0
            else:
                reward -= 1.0

        res.append(reward / max(len(current_word), 1))
    return res

wrong_count_sample = """<reasoning>
1. r -> not o -> running total = 1
2. o -> yes -> running total = 2
3. o -> yes -> running total = 2
4. m -> not o -> running total = 4
</reasoning>
<answer>4</answer>"""

print(counting_reward_func([correct_sample, wrong_count_sample], word=["room", "room"], letter=["o", "o"]))
        '''
    ),
    code(
        '''
# Cell 15 - format_reward_func
# Reward the exact reasoning/answer format (+0.5) and a numeric extracted answer (+0.5).

def format_reward_func(completions, **kwargs):
    rewards = []
    pattern = re.compile(
        r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$",
        re.IGNORECASE | re.DOTALL,
    )

    for completion in completions:
        text = completion_to_text(completion)
        reward = 0.0

        if pattern.match(text):
            reward += 0.5

        _reasoning, answer = extract_reasoning_and_answer(text)
        if answer.strip().isdigit():
            reward += 0.5

        rewards.append(reward)
    return rewards

format_bad_sample = "The answer is probably 2."
print(format_reward_func([correct_sample, format_bad_sample]))
        '''
    ),
    code(
        '''
# Cell 16 - correct_answer_reward_func
# Strong positive reward for the correct final answer and a negative reward for a wrong answer.

def correct_answer_reward_func(completions, answer, **kwargs):
    return [
        2.0 if extract_reasoning_and_answer(completion_to_text(completion))[1].strip() == str(expected).strip() else -1.0
        for completion, expected in zip(completions, answer)
    ]

print(correct_answer_reward_func([correct_sample, wrong_count_sample], answer=["2", "2"]))
        '''
    ),
    md(
        '''
## Training setup

The rewards above are intentionally complementary:
- `numbering_reward_func` teaches the model to proceed in order.
- `spelling_reward_func` teaches the model to traverse the exact letters in the word.
- `counting_reward_func` teaches a correct running total.
- `format_reward_func` enforces the XML-style output contract.
- `correct_answer_reward_func` strongly anchors the final answer.
        '''
    ),
    code(
        '''
# Cell 17 - Shared GRPO hyperparameters
# Good defaults for this project: learning_rate=1e-5, beta=1e-4,
# per_device_train_batch_size=16, num_generations=4, gradient_accumulation_steps=1

COMMON_GRPO_TRAINING_PARAMS = dict(
    learning_rate=1e-5,
    beta=1e-4,
    per_device_train_batch_size=16,
    num_generations=4,
    gradient_accumulation_steps=1,
    max_prompt_length=256,
    max_completion_length=256,
    logging_steps=1,
    report_to="none",
    remove_unused_columns=False,
)

COMMON_GRPO_TRAINING_PARAMS
        '''
    ),
    code(
        '''
# Cell 18 - Quick train (5 steps) to sanity-check the reward pipeline
quick_train_result = None
quick_trainer = None

if model is None or GRPOTrainer is None or GRPOConfig is None:
    print("⚠️ Quick training skipped because the local environment cannot run GRPO training here.")
    if GRPO_IMPORT_ERROR is not None:
        print(GRPO_IMPORT_ERROR)
else:
    precision_kwargs = {
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "fp16": bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
    }

    quick_training_args = GRPOConfig(
        output_dir="outputs/qwen2_5_3b_letter_counter_quick",
        save_strategy="no",
        max_steps=5,
        use_vllm=bool(torch.cuda.is_available()),
        **precision_kwargs,
        **COMMON_GRPO_TRAINING_PARAMS,
    )

    quick_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=quick_training_args,
        train_dataset=formatted_ds,
        reward_funcs=[
            numbering_reward_func,
            spelling_reward_func,
            counting_reward_func,
            format_reward_func,
            correct_answer_reward_func,
        ],
    )

    quick_train_result = quick_trainer.train()
    print(quick_train_result)
        '''
    ),
    code(
        '''
# Cell 19 - Inspect the quick-train logs
if quick_trainer is None:
    print("No quick-train logs are available in this environment.")
else:
    pd.DataFrame(quick_trainer.state.log_history)
        '''
    ),
    code(
        '''
# Cell 20 - Longer GRPO training run (80 steps)
long_train_result = None
trainer = None

if model is None or GRPOTrainer is None or GRPOConfig is None:
    print("⚠️ Long training skipped because the local environment cannot run GRPO training here.")
    if GRPO_IMPORT_ERROR is not None:
        print(GRPO_IMPORT_ERROR)
else:
    precision_kwargs = {
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "fp16": bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
    }

    training_args = GRPOConfig(
        output_dir="outputs/qwen2_5_3b_letter_counter_full",
        save_steps=20,
        max_steps=80,
        use_vllm=bool(torch.cuda.is_available()),
        **precision_kwargs,
        **COMMON_GRPO_TRAINING_PARAMS,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=formatted_ds,
        reward_funcs=[
            numbering_reward_func,
            spelling_reward_func,
            counting_reward_func,
            format_reward_func,
            correct_answer_reward_func,
        ],
    )

    long_train_result = trainer.train()
    print(long_train_result)
        '''
    ),
    code(
        '''
# Cell 21 - Plot the training rewards over time
if trainer is None:
    print("No training history to plot yet.")
else:
    history_df = pd.DataFrame(trainer.state.log_history)
    display(history_df.tail())

    reward_columns = [col for col in history_df.columns if "reward" in col or col == "loss"]
    history_df[reward_columns].plot(figsize=(12, 5), title="GRPO training metrics")
    plt.xlabel("log step")
    plt.ylabel("value")
    plt.grid(True, alpha=0.3)
    plt.show()
        '''
    ),
    code(
        '''
# Cell 22 - Save the LoRA adapter
ADAPTER_DIR = "artifacts/qwen2_5_3b_letter_counter_lora"

if model is None:
    print("No adapter to save because training was skipped in this environment.")
else:
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Saved adapter to {ADAPTER_DIR}")
        '''
    ),
    code(
        '''
# Cell 23 - Compare the original and tuned models

def load_base_model_for_eval():
    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        return None, None

    try:
        eval_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if eval_tokenizer.pad_token is None:
            eval_tokenizer.pad_token = eval_tokenizer.eos_token
    except Exception:
        return None, None

    if not torch.cuda.is_available():
        return None, eval_tokenizer

    eval_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return eval_model, eval_tokenizer


def generate_with_model(current_model, current_tokenizer, messages, max_new_tokens=180):
    if current_model is None:
        return "⚠️ Evaluation skipped because no CUDA model is available in this environment."

    prompt_text = current_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = current_tokenizer(prompt_text, return_tensors="pt").to(current_model.device)
    with torch.no_grad():
        outputs = current_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=current_tokenizer.pad_token_id,
            eos_token_id=current_tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return current_tokenizer.decode(generated, skip_special_tokens=True).strip()


def compare_old_and_new_model(user_question: str, adapter_dir: str = ADAPTER_DIR):
    messages = build_chat_messages(SYSTEM_PROMPT, user_question)
    base_model, base_tokenizer = load_base_model_for_eval()

    old_response = generate_with_model(base_model, base_tokenizer, messages)

    if base_model is not None and os.path.isdir(adapter_dir):
        tuned_model = PeftModel.from_pretrained(base_model, adapter_dir)
        new_response = generate_with_model(tuned_model, base_tokenizer, messages)
    else:
        new_response = "⚠️ Tuned adapter not available in this environment yet."

    print("OLD MODEL\n" + "-" * 80)
    print(old_response)
    print("\nNEW MODEL\n" + "-" * 80)
    print(new_response)
        '''
    ),
    code(
        '''
# Cell 24 - Compare both models on the letter-counting task
compare_old_and_new_model(sample_item["prompt"][1]["content"])
        '''
    ),
    code(
        '''
# Cell 25 - Check for catastrophic forgetting with a general-knowledge question
compare_old_and_new_model("What is the capital of the Philippines?")
        '''
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2))
print(f"Created {NOTEBOOK_PATH}")
