# GEN-AI2 Project Report

## Project Overview

This project focuses on improving a language model’s ability to perform **precise, procedural reasoning** on a narrow but important task: counting how many times a letter appears in a word.

The work was centered on the notebook `project/starter/gen_ai_fundamentals_project_starter.ipynb`, where the full pipeline was completed:
- prompt-engineering baseline,
- synthetic dataset creation,
- reward-function design,
- LoRA configuration,
- GRPO training setup,
- evaluation and reporting.

The base model selected was `Qwen/Qwen2.5-3B-Instruct`, and the planned fine-tuning strategy used **LoRA** for efficiency and **GRPO** for reinforcement learning.

---

## Dataset Description

The dataset is a **small synthetic reasoning dataset** designed specifically for the letter-counting task.

A list of sample words was created, including examples such as:
- `effectiveness`
- `banana`
- `committee`
- `bookkeeper`
- `mississippi`
- `coffee`
- `queue`
- `cheese`

For each word, records were generated for each distinct target letter in that word. Each record contains:
- the original `word`,
- the target `letter`,
- the correct final `answer`,
- a natural-language `user_prompt`.

### Verified local dataset size
The notebook generated **94 training records** during local execution.

---

## Data Preparation & Exploration

The data preparation process followed these steps:

1. A curated list of words was defined in `ALL_WORDS`.
2. A helper function, `generate_records`, expanded the words into prompt-answer records.
3. The records were converted into a Hugging Face `Dataset`.
4. Each example was formatted with the instruction prompt using `format_record`.
5. A sample item was inspected to verify the structure.

This preparation ensured that each training example matched the final chat-based prompting format expected by the model.

### Prompt structure
Each example was transformed into a conversation-like format with:
- a `system` message containing the reasoning instructions,
- a `user` message containing the counting question.

---

## Model Design

The model design used the following decisions:

### Base model
- `Qwen/Qwen2.5-3B-Instruct`

### Parameter-efficient fine-tuning
LoRA was selected to avoid full-model fine-tuning and to reduce memory costs.

### LoRA configuration
The notebook uses:
- `lora_rank = 64`
- target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

### Design rationale
- Rank `64` is a strong middle-ground value for a 3B model.
- The chosen modules cover the main attention and MLP projections, allowing the adapter to affect both token interactions and internal transformations.

---

## Training Process

The training plan followed the GRPO workflow described in the project:

1. **Baseline prompting** using a weak prompt.
2. **Improved prompting** using Chain-of-Thought plus a worked example.
3. **Reward engineering** to score model outputs.
4. **Quick GRPO training** for sanity checking.
5. **Longer GRPO training** for performance improvement.

### Reward functions implemented
The following reward functions were completed:
- `numbering_reward_func`
- `spelling_reward_func`
- `counting_reward_func`
- `format_reward_func`
- `correct_answer_reward_func`

### GRPO parameters used
The notebook includes the requested defaults:
- `learning_rate = 1e-5`
- `beta = 1e-4`
- `per_device_train_batch_size = 16`
- `num_generations = 4`
- `gradient_accumulation_steps = 1`

### Important local note
The notebook was prepared to run safely on this macOS workspace, but **full model training was not executed locally** because CUDA / NVIDIA GPU support is unavailable here.

---

## Model Evaluation

The evaluation design includes three layers:

### 1. Prompting baseline
The notebook first checks how the untuned model behaves with a minimal prompt.

### 2. Improved prompt baseline
A stronger system prompt with explicit reasoning steps and a worked example is then used to improve performance before RL training.

### 3. Post-training comparison
A helper function, `compare_old_and_new_model`, was added to compare:
- the original model,
- the tuned LoRA model.

A final general-knowledge check was also included to look for **catastrophic forgetting**.

---

## Results & Interpretation

### Verified local results
The lightweight notebook sections were executed successfully and produced the following reward-validation outputs:

- `numbering_reward_func` → `[2.0, -1.5]`
- `spelling_reward_func` → `[2.0, 0.10000000000000003]`
- `counting_reward_func` → `[1.0, -0.5]`
- `format_reward_func` → `[1.0, 0.0]`
- `correct_answer_reward_func` → `[2.0, -1.0]`

### Interpretation
These outputs show that the reward functions correctly distinguish better completions from worse ones, which is exactly what is needed before GRPO training.

### Environment limitation
The expensive training and full model-comparison stages were skipped locally because the verified environment output was:
- `Python version   : 3.12.3`
- `CUDA available   : False`

So the notebook is **functionally complete**, but the final adapter training must be run in a GPU-enabled environment.

---

## Non-Technical Explanation

In simple terms, this project teaches an AI model to **slow down and think carefully**.

Instead of guessing how many times a letter appears in a word, the model is encouraged to:
1. look at the word one letter at a time,
2. check whether each letter matches the target,
3. keep a correct count as it goes,
4. report the answer clearly.

This is useful because large language models are often fluent but not always reliable on exact counting or step-based logic.

---

## Experimental Design Justification

The experimental design is appropriate for this task for several reasons:

### Why a synthetic dataset?
The task is narrow, structured, and easy to generate automatically with correct labels.

### Why LoRA?
LoRA allows efficient fine-tuning without updating the full 3B-parameter model.

### Why GRPO?
GRPO is well suited to reasoning-style tasks where outputs can be judged by custom reward functions instead of plain next-token supervision.

### Why multiple rewards?
A single “correct final answer” reward is too weak on its own. The extra rewards encourage:
- orderly reasoning,
- exact spelling,
- accurate running counts,
- correct formatting.

This gives the model better intermediate guidance.

---

## Workflow Completeness

The workflow is complete at the notebook-design level:

- ✅ TODO sections were filled in.
- ✅ Hyperparameter choices were documented.
- ✅ Reward functions were implemented.
- ✅ Validation cells were executed for the lightweight sections.
- ✅ Comparison helpers were added.
- ✅ README documentation was updated.

### Remaining execution dependency
The only missing part is **full GPU-based training output**, which requires running the notebook in a proper CUDA environment such as:
- the course lab,
- Google Colab with a T4 GPU,
- a Linux machine with NVIDIA CUDA support.

---

## Bias and Risk Awareness

Although this is a narrow reasoning task, several risks still apply:

### 1. Reward hacking
A model may learn to exploit the reward structure without truly reasoning well.

### 2. Limited generalization
A small synthetic dataset may not fully generalize to unseen formatting or more complex spelling cases.

### 3. Overfitting to format
The model may become very good at output structure while still making occasional logical mistakes.

### 4. Environment mismatch
Results observed on macOS without GPU do not represent final training performance.

To reduce these risks, the notebook uses multiple complementary rewards and separates baseline prompting from RL fine-tuning.

---

## Future Improvements & Integration

Several improvements could strengthen the project:

1. **Expand the dataset** with more varied words, repeated letters, and harder edge cases.
2. **Add held-out evaluation data** for clearer generalization testing.
3. **Run full GRPO training on GPU** and save the final LoRA adapter.
4. **Track more metrics** such as exact match rate and step-level accuracy.
5. **Package the tuned model** behind a small API or demo app for interactive use.
6. **Test transfer** to related reasoning tasks like counting digits, vowels, or symbols.

---

## Conclusion

This project demonstrates a practical workflow for teaching a language model a narrow reasoning skill using:
- structured prompting,
- synthetic supervision,
- carefully designed rewards,
- and reinforcement learning with LoRA.

The notebook is complete and verified for the lightweight stages, and it is ready to be executed in a GPU-enabled environment for final GRPO fine-tuning and adapter generation.

---

## Reproducibility

To reproduce the local setup used during this work:

### Environment
```bash
uv python install 3.12.3
uv venv --python 3.12.3 .venv
uv pip install --python .venv/bin/python -r requirements.txt torch
```

### Launch notebook
```bash
source .venv/bin/activate
jupyter lab
```

Then open:
```text
project/starter/gen_ai_fundamentals_project_starter.ipynb
```

### Verified local environment state
- `uv 0.7.11`
- Python `3.12.3`
- No CUDA / `nvidia-smi` support in this workspace

For full reproducibility of the training section, use a **Linux CUDA environment with an NVIDIA GPU**.

---

## References

1. **Qwen2.5 Instruct models** — Hugging Face model documentation for `Qwen/Qwen2.5-3B-Instruct`
2. **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al.
3. **TRL documentation** — Hugging Face `trl` library for RLHF / GRPO workflows
4. **PEFT documentation** — Hugging Face parameter-efficient fine-tuning library
5. **Unsloth documentation** — accelerated fine-tuning utilities for LLMs
6. **vLLM documentation** — efficient inference engine for large language models
7. Course project instructions and rubric provided with the assignment
