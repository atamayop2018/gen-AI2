# GEN-AI2 — Letter-Counting Reasoning Project

This repository contains a completed notebook for a Generative AI fundamentals project focused on teaching an instruction-tuned LLM to perform **step-by-step reasoning** for a letter-counting task.

The core deliverable is:
- `project/starter/gen_ai_fundamentals_project_starter.ipynb`

---

## 📌 Project goal

The project fine-tunes `Qwen/Qwen2.5-3B-Instruct` with:
- **LoRA** for parameter-efficient adaptation
- **GRPO** for reinforcement learning
- custom reward functions for:
  - numbering
  - spelling
  - counting
  - output format
  - final-answer correctness

The target behavior is for the model to:
1. spell the word letter by letter,
2. check each letter against the target,
3. keep an accurate running count,
4. return the final answer in a structured format.

---

## 📂 Repository contents

| Path | Description |
|---|---|
| `project/starter/gen_ai_fundamentals_project_starter.ipynb` | Main project notebook |
| `project/starter/build_gen_ai_notebook.py` | Helper script used to generate the notebook scaffold |
| `requirements.txt` | Python dependencies |

---

## ✅ Notebook contents

The notebook includes:
- LoRA configuration with `lora_rank = 64`
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- a Chain-of-Thought system prompt with a worked example
- completed reward functions
- GRPO training parameter defaults
- model comparison and catastrophic-forgetting checks

---

## ⚙️ Local setup

This repo is set up to work with **Python 3.12.3**.

### 1) Create the environment

```bash
uv python install 3.12.3
uv venv --python 3.12.3 .venv
```

### 2) Install dependencies

```bash
uv pip install --python .venv/bin/python -r requirements.txt torch
```

### 3) Launch Jupyter

```bash
source .venv/bin/activate
jupyter lab
```

Then open:

```text
project/starter/gen_ai_fundamentals_project_starter.ipynb
```

and select the `.venv` / Python 3.12.3 kernel.

---

## ⚠️ GPU note

A **CUDA-enabled NVIDIA GPU** is required to actually:
- load the 3B model for training,
- run GRPO fine-tuning,
- save a real LoRA adapter such as `adapter_model.safetensors`.

In this macOS workspace, `nvidia-smi` is not available, so the notebook is designed to:
- run the setup and reward-validation cells,
- skip heavy training cells safely,
- show clear status messages when GPU-only steps are unavailable.

For full training, run the notebook in the course GPU lab, Google Colab with a T4 GPU, or another Linux CUDA environment.

---

## 🔎 Local verification status

The notebook has been verified locally for the lightweight sections:
- Python version: `3.12.3`
- GPU availability: `False`
- reward validation outputs observed:
  - numbering: `[2.0, -1.5]`
  - spelling: `[2.0, 0.10000000000000003]`
  - counting: `[1.0, -0.5]`
  - format: `[1.0, 0.0]`
  - correct answer: `[2.0, -1.0]`

---

## 🚀 Next step

To complete the full assignment submission, run the later training and evaluation cells in a GPU-enabled environment so the notebook shows:
- training logs,
- reward trends,
- saved adapter artifacts,
- tuned-model comparison outputs.