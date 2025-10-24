import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple

import torch


@dataclass
class ResidualsConfig:
    residuals_version: str
    format: str
    dtype: str
    param_count: int
    shapes_hash: str
    parameter_names_hash: str
    tokenizer_name: Optional[str]
    created_at: str
    library: Dict[str, str]
    base_model_name: Optional[str] = None
    instruct_model_name: Optional[str] = None
    pipeline_tag: Optional[str] = "text-generation"
    license: Optional[str] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None
    base_model_relation: Optional[str] = "adapter"


def build_config(
    residuals: Dict[str, torch.Tensor],
    tokenizer_name: Optional[str],
    base_model_name: Optional[str] = None,
    instruct_model_name: Optional[str] = None,
) -> ResidualsConfig:
    names = sorted(residuals.keys())
    shapes: List[Tuple[int, ...]] = [tuple(residuals[n].shape) for n in names]
    names_hash = hashlib.sha256("|".join(names).encode()).hexdigest()
    shapes_hash = hashlib.sha256(
        "|".join([f"{n}:{list(s)}" for n, s in zip(names, shapes)]).encode()
    ).hexdigest()
    dtypes = [str(t.dtype).split(".")[-1] for t in residuals.values()]
    dtype = max(set(dtypes), key=dtypes.count) if dtypes else "float32"
    return ResidualsConfig(
        residuals_version="1.0",
        format="pytorch",
        dtype=dtype,
        param_count=len(residuals),
        shapes_hash=shapes_hash,
        parameter_names_hash=names_hash,
        tokenizer_name=tokenizer_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        library={"package": "residuals"},
        base_model_name=base_model_name,
        instruct_model_name=instruct_model_name,
        pipeline_tag="text-generation",
        tags=["residuals", "delta", "task-arithmetic", "finetune"],
        base_model_relation="adapter",
    )
