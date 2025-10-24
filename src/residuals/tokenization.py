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
