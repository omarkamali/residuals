from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer


def align_tokenizer_and_embeddings(
    base_model: AutoModelForCausalLM, base_tokenizer: AutoTokenizer
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Align tokenizer and embeddings by adding PAD token if missing and zeroing new rows.
    """
    DEFAULT_PAD_TOKEN = "[PAD]"
    num_new = 0

    if base_tokenizer.pad_token is None:
        num_new = base_tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))

    if num_new > 0:
        base_model.resize_token_embeddings(len(base_tokenizer))

        # Zero out new embedding rows to avoid random initialization
        input_embeddings = base_model.get_input_embeddings().weight.data
        output_embeddings = base_model.get_output_embeddings().weight.data

        input_embeddings[-num_new:] = 0
        output_embeddings[-num_new:] = 0

    return base_model, base_tokenizer


def resize_model_to_tokenizer(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer
) -> AutoModelForCausalLM:
    """
    Resize model embeddings (and output head if tied) to match tokenizer length.
    Zero-initialize any newly added rows.
    """
    target = len(tokenizer)
    current = model.get_input_embeddings().weight.data.shape[0]
    if target == current:
        return model
    if target < current:
        # Do not shrink automatically; caller must decide how to handle removal
        raise ValueError(f"Cannot shrink model vocab from {current} to {target} automatically")
    # Grow
    model.resize_token_embeddings(target)
    num_new = target - current
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    input_embeddings[-num_new:] = 0
    output_embeddings[-num_new:] = 0
    return model
