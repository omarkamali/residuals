from typing import Optional


def infer_model_name(model) -> Optional[str]:
    try:
        return getattr(model.config, "name_or_path", None) or getattr(model.config, "_name_or_path", None)
    except Exception:
        return None


def infer_tokenizer_name(tokenizer) -> Optional[str]:
    try:
        return getattr(tokenizer, "name_or_path", None)
    except Exception:
        return None
