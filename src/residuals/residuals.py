"""
Instruction Residuals for LLM Continuous Pre-Training

Based on "Balancing Continuous Pre-Training and Instruction Fine-Tuning" 
(Samsung Research, 2024) and "Editing Models with Task Arithmetic" (Ilharco et al., 2022)

Key findings from the literature:
- Instruction capabilities are portable across models from the same family
- Simple element-wise addition (task arithmetic) is effective for merging
- No additional scaling or normalization is typically needed
- Tokenizer alignment is critical before applying residuals
"""

import json
import os
from dataclasses import asdict
from typing import Dict, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import ResidualsConfig, build_config
from .metadata import infer_model_name, infer_tokenizer_name
from .readme import build_readme
from .tokenization import align_tokenizer_and_embeddings

# Optional Hub imports at module level for easier monkeypatching in tests
try:  # pragma: no cover - handled in tests via monkeypatch
    from huggingface_hub import HfApi, upload_folder, snapshot_download
except Exception:  # pragma: no cover
    HfApi = None  # type: ignore
    upload_folder = None  # type: ignore
    snapshot_download = None  # type: ignore

def _download_from_hub(repo_id: str, token: Optional[str] = None) -> str:
    """Download a repo snapshot and return the local directory path.

    Split to a helper for easier testing via monkeypatching.
    """
    if repo_id.startswith("hf://"):
        # strip optional prefix
        repo_id = repo_id[len("hf://") :]
    if snapshot_download is None:
        raise ImportError("huggingface_hub is required to download from hub. Install huggingface_hub.")
    return snapshot_download(repo_id=repo_id, token=token)


# ResidualsConfig moved to config.py


class Residuals:
    """Container for instruction residuals with load/save/apply ergonomics."""

    def __init__(self, state_dict: Dict[str, torch.Tensor], config: ResidualsConfig, instruct_tokenizer: Optional[AutoTokenizer] = None):
        self.state_dict = state_dict
        self.config = config
        self.instruct_tokenizer = instruct_tokenizer

    @staticmethod
    def from_models(
        base_model: Optional[AutoModelForCausalLM] = None,
        instruct_model: Optional[AutoModelForCausalLM] = None,
        base_model_name: Optional[str] = None,
        instruct_model_name: Optional[str] = None,
        instruct_tokenizer: Optional[AutoTokenizer] = None,
        instruct_tokenizer_name: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = "cpu",
    ) -> "Residuals":
        # Allow passing either models or names; load if names were provided
        if base_model is None:
            if base_model_name is None:
                raise ValueError("Either base_model or base_model_name must be provided")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=dtype
            )
            # Place on requested device if provided
            try:
                if device is not None:
                    base_model = base_model.to(device)
                base_model = base_model.to(dtype=dtype)
            except Exception:
                pass
        if instruct_model is None:
            if instruct_model_name is None:
                raise ValueError("Either instruct_model or instruct_model_name must be provided")
            instruct_model = AutoModelForCausalLM.from_pretrained(
                instruct_model_name, torch_dtype=dtype
            )
            try:
                if device is not None:
                    instruct_model = instruct_model.to(device)
                instruct_model = instruct_model.to(dtype=dtype)
            except Exception:
                pass

        base_sd = base_model.state_dict()
        inst_sd = instruct_model.state_dict()
        state: Dict[str, torch.Tensor] = {}
        for key, base_param in base_sd.items():
            if key not in inst_sd:
                raise KeyError(f"Missing key in instruct model: {key}")
            inst_param = inst_sd[key]
            if base_param.shape != inst_param.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: base {base_param.shape} vs instruct {inst_param.shape}"
                )
            state[key] = (inst_param - base_param).to(dtype=inst_param.dtype)
        # Infer names from instances if not provided
        if base_model_name is None and base_model is not None:
            base_model_name = infer_model_name(base_model)
        if instruct_model_name is None and instruct_model is not None:
            instruct_model_name = infer_model_name(instruct_model)

        # Determine instruct tokenizer
        tok = instruct_tokenizer
        if tok is None and instruct_tokenizer_name is not None:
            tok = AutoTokenizer.from_pretrained(instruct_tokenizer_name, use_fast=False)
        if instruct_tokenizer_name is None and tok is not None:
            instruct_tokenizer_name = infer_tokenizer_name(tok)

        tok_name = instruct_tokenizer_name
        cfg = build_config(
            state,
            tokenizer_name=tok_name,
            base_model_name=base_model_name,
            instruct_model_name=instruct_model_name,
        )
        return Residuals(state, cfg, instruct_tokenizer=tok)

    @staticmethod
    def from_pretrained(
        path: str,
        map_location: Union[str, torch.device] = "cpu",
        token: Optional[str] = None,
    ) -> "Residuals":
        # If path looks like a Hub repo ID (or non-existent path), download it first
        resolved_path = path
        if not os.path.isdir(path):
            if "/" in path or path.startswith("hf://"):
                resolved_path = _download_from_hub(path, token=token)

        # Prefer safetensors (single or sharded)
        index_path = os.path.join(resolved_path, "model.safetensors.index.json")
        st_path_st = os.path.join(resolved_path, "model.safetensors")
        state: Dict[str, torch.Tensor] = {}
        if os.path.exists(index_path):
            import json as _json
            from safetensors.torch import load_file as st_load
            with open(index_path, "r", encoding="utf-8") as f:
                idx = _json.load(f)
            weight_map = idx.get("weight_map", {})
            shard_files = sorted(set(weight_map.values()))
            for shard in shard_files:
                shard_path = os.path.join(resolved_path, shard)
                if not os.path.exists(shard_path):
                    raise FileNotFoundError(f"Shard file missing: {shard_path}")
                tensors = st_load(shard_path, device=map_location)
                state.update(tensors)
        elif os.path.exists(st_path_st):
            from safetensors.torch import load_file as st_load
            state = st_load(st_path_st, device=map_location)
        else:
            raise FileNotFoundError("model.safetensors or model.safetensors.index.json not found")

        cfg_path = os.path.join(resolved_path, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            cfg = ResidualsConfig(**raw)
        else:
            raise FileNotFoundError("config.json not found alongside residuals; tokenizer required by current format")

        # Load tokenizer saved in the same directory
        tok = AutoTokenizer.from_pretrained(resolved_path, use_fast=False)
        return Residuals(state, cfg, instruct_tokenizer=tok)

    def apply(
        self,
        base_model: AutoModelForCausalLM,
        base_tokenizer: Optional[AutoTokenizer] = None,
        instruct_tokenizer: Optional[AutoTokenizer] = None,
        out_dir: Optional[str] = None,
        scale: float = 1.0,
    ) -> AutoModelForCausalLM:
        # Prefer provided instruct_tokenizer; else fall back to stored one
        if instruct_tokenizer is None and self.instruct_tokenizer is not None:
            instruct_tokenizer = self.instruct_tokenizer

        if base_tokenizer is not None and instruct_tokenizer is not None:
            base_model, base_tokenizer = align_tokenizer_and_embeddings(
                base_model, base_tokenizer
            )

        groups: Dict[int, Dict[str, torch.Tensor]] = {}
        tensors_by_ptr: Dict[int, torch.Tensor] = {}

        for n, p in base_model.named_parameters():
            if n not in self.state_dict:
                raise KeyError(f"Residuals missing key: {n}")
            delta = self.state_dict[n]
            if delta.shape != p.shape:
                raise ValueError(
                    f"Shape mismatch for {n}: param {p.shape} vs delta {delta.shape}"
                )
            try:
                ptr = p.data_ptr()
            except Exception:
                ptr = id(p)
            if ptr not in groups:
                groups[ptr] = {}
                tensors_by_ptr[ptr] = p
            groups[ptr][n] = delta

        with torch.no_grad():
            for ptr, name_to_delta in groups.items():
                t = tensors_by_ptr[ptr]
                total_delta = None
                for d in name_to_delta.values():
                    d_ = (d * scale).to(t.dtype)
                    total_delta = d_ if total_delta is None else (total_delta + d_)
                t.add_(total_delta)

        for n, b in base_model.named_buffers():
            if n in self.state_dict:
                d = self.state_dict[n]
                if d.shape != b.shape:
                    raise ValueError(
                        f"Shape mismatch for buffer {n}: param {b.shape} vs delta {d.shape}"
                    )
                with torch.no_grad():
                    b.add_((d * scale).to(b.dtype))

        if out_dir:
            base_model.save_pretrained(out_dir)
            if instruct_tokenizer is not None:
                instruct_tokenizer.save_pretrained(out_dir)

        return base_model

    def save_pretrained(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # Save as safetensors for HF compatibility
        from safetensors.torch import save_file as st_save
        st_save(self.state_dict, os.path.join(out_dir, "model.safetensors"))
        # Always require tokenizer to be saved
        if self.instruct_tokenizer is None:
            raise ValueError("instruct_tokenizer must be provided to save_pretrained; set it via from_models(... instruct_tokenizer=... or instruct_tokenizer_name=...)")
        self.instruct_tokenizer.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)
        readme_path = os.path.join(out_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(build_readme(self.config))

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Residuals":
        """Return a new Residuals instance with tensors moved/cast to device/dtype.

        Does not modify the original instance.
        """
        new_state: Dict[str, torch.Tensor] = {}
        for k, v in self.state_dict.items():
            new_state[k] = v.to(device=device if device is not None else v.device, dtype=dtype if dtype is not None else v.dtype)

        # Update config dtype string based on new tensors
        names = sorted(new_state.keys())
        dtypes = [str(new_state[n].dtype).split(".")[-1] for n in names]
        dtype_str = max(set(dtypes), key=dtypes.count) if dtypes else "float32"
        new_cfg = ResidualsConfig(
            residuals_version=self.config.residuals_version,
            format=self.config.format,
            dtype=dtype_str,
            param_count=self.config.param_count,
            shapes_hash=self.config.shapes_hash,
            parameter_names_hash=self.config.parameter_names_hash,
            tokenizer_name=self.config.tokenizer_name,
            created_at=self.config.created_at,
            library=self.config.library,
        )
        return Residuals(new_state, new_cfg, instruct_tokenizer=self.instruct_tokenizer)

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        exist_ok: bool = True,
        token: Optional[str] = None,
    ) -> str:
        """Push residuals to the Hugging Face Hub.

        Saves artifacts to a temporary directory, creates the repo if needed, and uploads
        all files. Returns the repo URL.
        """
        import tempfile
        if HfApi is None or upload_folder is None:
            raise ImportError("huggingface_hub is required to push to hub. Install huggingface_hub.")

        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir)

            api = HfApi()
            api.create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=exist_ok,
                token=token,
            )
            upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                token=token,
                path_in_repo="",
            )
        return f"https://huggingface.co/{repo_id}"


    def _download_from_hub(repo_id: str, token: Optional[str] = None) -> str:
        """Download a repo snapshot and return the local directory path.

        Split to a helper for easier testing via monkeypatching.
        """
        if repo_id.startswith("hf://"):
            # strip optional prefix
            repo_id = repo_id[len("hf://") :]
        if snapshot_download is None:
            raise ImportError("huggingface_hub is required to download from hub. Install huggingface_hub.")
        return snapshot_download(repo_id=repo_id, token=token)

