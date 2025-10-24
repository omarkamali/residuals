# Instruction Residuals

[![PyPI version](https://badge.fury.io/py/residuals.svg)](https://badge.fury.io/py/residuals)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight Python package implementing **instruction residuals** (task vectors) for efficient LLM continuous pre-training, based on the methodology from Samsung Research's 2024 paper and the task arithmetic paradigm.

## Overview

Extract instruction capabilities from instruction-tuned models, continue pre-training on domain data, then instantly restore instruction-following abilities—**~2000x more compute-efficient** than full instruction fine-tuning.

### Key Benefits

- **Instruction capabilities are portable** across models from the same family  
- **CPT on instruction models causes catastrophic forgetting** of instruction abilities which this technique mitigates  
- **CPT on base models** preserves knowledge when residuals are reapplied to regain SFT capabilities 
- **No additional instruction tuning needed** after CPT  
- **~2000x more compute-efficient** than full instruction fine-tuning

## Installation

### Using uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project with residuals
uv init my-cpt-project
cd my-cpt-project
uv add residuals

# Or add to existing project
uv add residuals
```

### Using pip

```bash
pip install residuals
```

### From source

```bash
git clone https://github.com/omarkamali/residuals.git
cd residuals
uv pip install -e .
```

## Quick Start

### Complete Workflow: CPT → Residual Application → SFT

#### 1. Compute and Save Instruction Residuals (Once)

```python
from residuals import Residuals
from transformers import AutoModelForCausalLM
import torch

# Paths to your base and instruction-tuned models
base_path = "meta-llama/Meta-Llama-3-8B"
instruct_path = "meta-llama/Meta-Llama-3-8B-Instruct"
delta_out = "./llama3_instruct_residuals"

# Compute residuals (Θ_r = θ_instruct - θ_base) and persist tokenizer
res = Residuals.from_models(
    base_model_name=base_path,
    instruct_model_name=instruct_path,
    instruct_tokenizer_name=instruct_path,
    dtype=torch.float32,
)
res.save_pretrained(delta_out)
```

**Key Finding**: Residuals computed from LLaMA 3.1 can improve LLaMA 3 base models, demonstrating cross-version portability.

#### 2. Continuous Pre-Training on Base Model

```python
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# Load BASE model for CPT (not instruction model!)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_path,
    max_seq_length=4096,
    load_in_4bit=True,
)

# Load domain corpus
dataset = load_dataset("text", data_files={"train": "domain_corpus.txt"})["train"]

# CPT with SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        max_steps=5000,
        learning_rate=2e-4,
        output_dir="outputs_cpt",
    ),
)

trainer.train()
model.save_pretrained_merged("ckpts/base_cpt_fp16", tokenizer, save_method="merged_16bit")
```

**Why CPT the base?** Samsung paper shows CPT on instruction models loses instruction capabilities, requiring expensive re-tuning.

#### 3. Reapply Instruction Residuals to CPT'd Base

```python
from residuals import Residuals
from transformers import AutoModelForCausalLM
import torch

# Load CPT'd base
cpt_model = AutoModelForCausalLM.from_pretrained("ckpts/base_cpt_fp16", dtype=torch.float32)

# Load saved residuals (tokenizer loaded from the same directory)
res = Residuals.from_pretrained(delta_out)

# Apply via element-wise addition
res.apply(
    base_model=cpt_model,
    out_dir="ckpts/base_cpt_plus_instruct"
)
```

**Result**: Your model now has both domain knowledge from CPT AND instruction-following capabilities—with ~2000x less compute than full instruction tuning.

#### 4. (Optional) Task-Specific SFT

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ckpts/base_cpt_plus_instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
)

# ... train with SFTTrainer on task-specific data
model.save_pretrained_merged("ckpts/final_model", tokenizer, save_method="merged_16bit")
```


### GPU acceleration (optional)

If you want faster residual computation/application on large models, install the optional GPU extras and set the device explicitly:

```bash
pip install -e .[gpu]
```

Then use `device="cuda"` when creating residuals from model names (instances you pass in are respected as-is):

```python
from residuals import Residuals

res = Residuals.from_models(
    base_model_name="meta-llama/Meta-Llama-3-8B",
    instruct_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    instruct_tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    device="cuda",
)
```

### Adjusting device/dtype after computing residuals

You can cast or move residual tensors after computation using `.to(device=..., dtype=...)`:

```python
from residuals import Residuals
from transformers import AutoModelForCausalLM
import torch

# Compute on CPU
res = Residuals.from_models(
    base_model_name="meta-llama/Meta-Llama-3-8B",
    instruct_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    instruct_tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
)

# Optionally cast/move residuals
res_fp16 = res.to(dtype=torch.float16)            # cast to fp16
# res_cuda = res.to(device="cuda", dtype=torch.float16)  # move and cast (requires GPU extras)

base = AutoModelForCausalLM.from_pretrained("ckpts/base_cpt_fp16", dtype=torch.float32)
res_fp16.apply(base, out_dir="ckpts/base_cpt_plus_instruct")
```

## Mathematical Foundation

**Instruction Residuals** (Equation 1 from Samsung paper):
```
Θ_r = θ_instruct - θ_base
```

**Application via Task Arithmetic** (Equation 2):
```
θ_cpt_instruct = θ_cpt_base ⊕ Θ_r
```

Where `⊕` represents element-wise addition, following the task arithmetic paradigm (Ilharco et al., 2022).

## Implementation Details

### No Scaling Needed for Same-Family Models

Samsung paper empirically demonstrates that when applying residuals within the same model family (e.g., LLaMA 3 → 3.1), **no scaling factor is required**. Element-wise addition works directly.

### Tokenizer Alignment

The implementation automatically:
1. Checks if base tokenizer lacks a PAD token
2. Adds PAD token if missing (`[PAD]`)
3. Resizes embeddings to match vocabulary
4. **Zeros newly added embedding rows** to prevent contamination

### Cross-Family Portability

Samsung paper (Table 3) shows:
- LLaMA 3.1 residuals → LLaMA 3 base: **better than LLaMA 3 instruct**
- LLaMA 3 residuals → LLaMA 3.1 base: improves over base, slightly below 3.1 instruct
- Works across Qwen 2 ↔ 2.5 families

Higher-quality instruct models produce better residuals.

## Model Card Auto-Generation (README.md)

When you call `Residuals.save_pretrained(out_dir)`, a Hugging Face-ready `README.md` is automatically generated in `out_dir` with:

- **Front-matter** including lineage and metadata:
  - `base_model`: the base model repo ID
  - `base_model_relation: adapter`
  - `instruct_model`: the instruction-tuned model repo ID
  - `pipeline_tag`, `tags`, `license`, `language`, and `library_name`
- **Usage** section showing how to load and apply the residuals
- **Files** and **Provenance** sections with hashes and creation info
- **Tools** section referencing the PyPI package `residuals`

Lineage fields are inferred even if you pass only model/tokenizer instances (no names):

```python
from residuals import Residuals
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
inst = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

res = Residuals.from_models(base_model=base, instruct_model=inst, instruct_tokenizer=tok)
res.save_pretrained("./llama3_instruct_residuals")  # writes README.md with lineage
```

You can optionally set additional metadata before saving:

```python
res.config.license = "apache-2.0"
res.config.language = "en"
res.config.tags = ["residuals", "llama", "finetune"]
```

Under the hood, this behavior lives in:

- `src/residuals/config.py`: `ResidualsConfig` dataclass
- `src/residuals/metadata.py`: model/tokenizer name inference
- `src/residuals/readme.py`: front-matter and README builders

## Push to Hugging Face Hub

You can push residuals to the Hub with one line. This publishes `model.safetensors`, `config.json`, tokenizer files, and an auto-generated `README.md`.

```python
from residuals import Residuals

# ... compute or load residuals into `res`
res.push_to_hub(
    repo_id="your-username/llama3-8b-instruct-residuals",
    private=True,   # set False to make public
    token="hf_..."  # or rely on local HF auth
)
```

Loading from the Hub is symmetric and compatible with `Residuals.from_pretrained()`:

```python
from residuals import Residuals

res = Residuals.from_pretrained("your-username/llama3-8b-instruct-residuals")
# If private, provide token:
# res = Residuals.from_pretrained("your-username/llama3-8b-instruct-residuals", token="hf_...")
```


## When to Use

✅ **Use instruction residuals when:**
- You want to CPT a model on domain-specific data
- Original base + instruct models are available
- You need compute efficiency (no instruction tuning budget)
- Working within the same model family

❌ **Limitations:**
- Requires both base and instruct models initially
- Best for same-family models (cross-family may degrade)
- Smaller models (<1.5B) show higher variance

## Testing

```bash
# With uv
uv run pytest

# With pip
pytest
```

## Development

```bash
# Clone repository
git clone https://github.com/omarkamali/residuals.git
cd residuals

# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests with coverage
uv run pytest --cov=residuals --cov-report=html

# Format code
uv run ruff format .

# Lint
uv run ruff check .
```

## References

1. **Jindal et al. (2024)** - "Balancing Continuous Pre-Training and Instruction Fine-Tuning" ([arXiv:2410.10739](https://arxiv.org/abs/2410.10739))
   - Introduces instruction residuals for LLMs
   - ~2000x compute savings vs. instruction tuning

2. **Ilharco et al. (2022)** - "Editing Models with Task Arithmetic" ([arXiv:2212.04089](https://arxiv.org/abs/2212.04089))
   - Foundational work on task vectors
   - Shows task vectors can be added/subtracted

3. **Yadav et al. (2023)** - "TIES-Merging" ([arXiv:2306.01708](https://arxiv.org/abs/2306.01708))
   - Advanced merging techniques for conflicts

4. **Community Implementations**:
   - Stanford Alpaca `weight_diff.py`
   - Vicuna/LLaVA/StableVicuna `apply_delta.py`

## License

MIT License - see [LICENSE](LICENSE) file

## Citation

If you use this package in your research, please cite:

```bibtex
@software{residuals2025,
  author = {Kamali, Omar},
  title = {Residuals: Instruction Residuals for Efficient LLM CPT},
  year = {2025},
  url = {https://github.com/omarkamali/residuals}
}

@article{jindal2024balancing,
  title={Balancing Continuous Pre-Training and Instruction Fine-Tuning},
  author={Jindal, Ishan and others},
  journal={arXiv preprint arXiv:2410.10739},
  year={2024}
}
```

## Contributing

Contributions welcome! Please open issues or PRs on GitHub.

**Maintained by**: [Omar Kamali](https://pypi.org/user/omarkamali/)  
**Contact**: residuals@omarkama.li
