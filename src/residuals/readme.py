from typing import List

from .config import ResidualsConfig


def build_front_matter(cfg: ResidualsConfig) -> str:
    tags: List[str] = cfg.tags or ["residuals", "delta", "task-arithmetic", "finetune"]
    lines: List[str] = ["---"]

    def emit(key: str, value):
        if value is None:
            return
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"- {item}")
        else:
            lines.append(f"{key}: {value}")

    emit("library_name", (cfg.library or {}).get("package", "residuals"))
    emit("base_model", cfg.base_model_name)
    emit("base_model_relation", cfg.base_model_relation or "adapter")
    emit("instruct_model", cfg.instruct_model_name)
    emit("pipeline_tag", cfg.pipeline_tag or "text-generation")
    emit("license", cfg.license)
    emit("language", [cfg.language] if isinstance(cfg.language, str) else cfg.language)
    emit("tags", tags)

    lines.append("---")
    return "\n".join(lines) + "\n"


essential_sections = """
## Files
- **model.safetensors**: Serialized residual tensors (safetensors format).
- (optional) **model.safetensors.index.json** + shard files `model-00001-of-000N.safetensors`, ... for multi-part weights.
- **config.json**: Residuals metadata and provenance.
- **tokenizer files**: Saved tokenizer for compatibility.

## About this format
These are additive residuals (task vectors). Applying them to the base model's parameters reconstructs the instruction-tuned model.

## Tools
Generated with the `residuals` Python package. Install via: `pip install residuals`.
- PyPI: https://pypi.org/project/residuals/
- Source: https://github.com/omarish/residuals
""".strip()


def build_readme(cfg: ResidualsConfig) -> str:
    fm = build_front_matter(cfg)

    lines: List[str] = [fm]
    lines.append("# Instruction Residuals")

    if cfg.base_model_name and cfg.instruct_model_name:
        lines.append(
            f"This repository contains instruction residuals (delta weights) computed as the parameter-wise difference between `{cfg.instruct_model_name}` and `{cfg.base_model_name}`."
        )
    else:
        lines.append(
            "This repository contains instruction residuals (delta weights) computed as the parameter-wise difference between an instruction-tuned model and its base model."
        )
    lines.append(
        "Apply these residuals to the base model to reconstruct the instruction-tuned weights without retraining."
    )

    lines.append("\n## Usage")
    py_base = cfg.base_model_name or "<base-model>"
    usage = f"""
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from residuals import Residuals

base = AutoModelForCausalLM.from_pretrained("{py_base}")
tok = AutoTokenizer.from_pretrained("{py_base}")

res = Residuals.from_pretrained(".")
res.apply(base, base_tokenizer=tok)
```"""
    lines.append(usage.strip())

    # Provenance
    prov = [
        f"- **Created at**: {cfg.created_at}",
        f"- **DType**: {cfg.dtype}",
        f"- **Parameters**: {cfg.param_count}",
        f"- **Shapes hash**: {cfg.shapes_hash}",
        f"- **Names hash**: {cfg.parameter_names_hash}",
    ]
    if cfg.base_model_name:
        prov.append(f"- **Base model**: `{cfg.base_model_name}`")
    if cfg.instruct_model_name:
        prov.append(f"- **Instruction model**: `{cfg.instruct_model_name}`")

    lines.append("\n## Provenance\n" + "\n".join(prov))
    lines.append(essential_sections)

    return "\n\n".join(lines) + "\n"
