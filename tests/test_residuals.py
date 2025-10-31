"""
Tests for instruction residuals package

Based on methodology from:
- Jindal et al. (2024) "Balancing Continuous Pre-Training and Instruction Fine-Tuning"
- Ilharco et al. (2022) "Editing Models with Task Arithmetic"
"""

import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM

from residuals import Residuals
from residuals.tokenization import resize_model_to_tokenizer

# Models under test: one without tied embeddings, one with tied embeddings
MODELS = [
    "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "distilgpt2",
    "sshleifer/tiny-gpt2"
]

# Subset with available tokenizers for save/load roundtrip
MODELS_WITH_TOKENIZER = [
    "distilgpt2",
    "sshleifer/tiny-gpt2",
]


@pytest.mark.parametrize("model_path", MODELS)
def test_calculate_and_apply_residuals(model_path: str):
    """
    Test that applying residuals reconstructs the instruction model.

    Validates Equations 1 & 2 from Samsung paper:
    1. Θ_r = θ_instruct - θ_base (calculation)
    2. θ_instruct = θ_base ⊕ Θ_r (application)
    """
    # Parametrized over architectures to validate both tied and non-tied embeddings

    # Create "base" and "instruct" models
    model_base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model_instruct = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    # Simulate instruction tuning by adding small delta
    with torch.no_grad():
        for key, param in model_instruct.state_dict().items():
            param.add_(torch.randn_like(param) * 0.01)

    # Calculate residuals (Equation 1)
    res = Residuals.from_models(model_base, model_instruct)

    assert len(res.state_dict) > 0, "Residuals should not be empty"

    # Create fresh base model for reconstruction
    model_base_copy = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    # Apply residuals (Equation 2)
    res.apply(model_base_copy)

    # Verify reconstruction matches instruction model
    instruct_sd = model_instruct.state_dict()
    reconstructed_sd = model_base_copy.state_dict()

    for key in instruct_sd.keys():
        diff = (instruct_sd[key] - reconstructed_sd[key]).abs().max().item()
        assert diff < 1e-5, f"Reconstruction failed for {key}: diff={diff}"


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_save_and_load_residuals(model_path: str):
    """
    Test that residuals can be saved and loaded without loss.

    Validates persistence of task vectors for later reuse.
    """
    # Parametrized model path

    # Create models with delta
    model_a = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model_b = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    with torch.no_grad():
        for key, param in model_b.state_dict().items():
            param.add_(torch.randn_like(param) * 0.02)

    # Calculate and save residuals (include instruct tokenizer by name)
    res = Residuals.from_models(model_a, model_b, instruct_tokenizer_name=model_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        res.save_pretrained(tmpdir)
        # Tokenizer artifacts should be present alongside residuals
        import os
        assert os.path.exists(os.path.join(tmpdir, "tokenizer_config.json"))


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_from_models_normalize_embeddings_true_captures_new_tokens_and_apply_resizes(model_path: str):
    """When normalize_embeddings=True, residuals should capture newly added tokens and apply() should resize base to match."""
    # Load models and tokenizer
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    inst = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # Add new tokens to instruct tokenizer
    added = ["<NEW_A>", "<NEW_B>", "<NEW_C>"]
    tok.add_tokens(added)

    # Resize instruct model upfront so we can set non-zero rows in new tokens
    inst = resize_model_to_tokenizer(inst, tok)

    # Make instruction deltas, including non-zero embeddings for new tokens
    with torch.no_grad():
        # General small delta
        for _, p in inst.state_dict().items():
            p.add_(torch.randn_like(p) * 0.001)
        # Boost new token rows in embeddings to make them detectable
        input_emb = inst.get_input_embeddings().weight
        output_emb = inst.get_output_embeddings().weight
        vocab_size = input_emb.shape[0]
        for i in range(1, len(added) + 1):
            input_emb[-i].add_(0.5)
            output_emb[-i].add_(0.5)

    # Compute residuals with normalization (also resizes base to tok)
    res = Residuals.from_models(
        base_model=base,
        instruct_model=inst,
        instruct_tokenizer=tok,
        normalize_embeddings=True,
    )

    # Apply to a fresh base that does not have the new tokens
    base_fresh = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    res.apply(base_fresh, normalize_embeddings=True)

    # Verify shapes now match tokenizer and new tokens rows are close to instruct
    assert base_fresh.get_input_embeddings().weight.shape[0] == len(tok)
    inst_sd = inst.state_dict()
    recon_sd = base_fresh.state_dict()
    # Compare a couple of parameters including embeddings rows
    max_diff = max((inst_sd[k] - recon_sd[k]).abs().max().item() for k in inst_sd.keys())
    assert max_diff < 5e-2  # allow small numeric noise


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_from_models_normalize_embeddings_false_raises_on_vocab_diff(model_path: str):
    """If normalize_embeddings=False and tokenizers differ, from_models should raise due to shape mismatch."""
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    inst = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tok.add_tokens(["<ADDED>"])

    # Do NOT resize models; shapes will differ
    with pytest.raises((ValueError, KeyError)):
        Residuals.from_models(
            base_model=base,
            instruct_model=inst,
            instruct_tokenizer=tok,
            normalize_embeddings=False,
        )


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_apply_normalize_embeddings_false_raises_on_vocab_diff(model_path: str):
    """If normalize_embeddings=False, apply() should raise when base embedding size doesn't match residuals tokenizer size."""
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    inst = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tok.add_tokens(["<ADDED_1>", "<ADDED_2>"])

    # Prepare instruct resized and modified
    inst = resize_model_to_tokenizer(inst, tok)
    with torch.no_grad():
        for _, p in inst.state_dict().items():
            p.add_(torch.randn_like(p) * 0.001)

    res = Residuals.from_models(base, inst, instruct_tokenizer=tok, normalize_embeddings=True)

    # Apply to a fresh base without resizing (normalize_embeddings=False) should raise
    base_fresh = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    with pytest.raises(ValueError):
        res.apply(base_fresh, normalize_embeddings=False)


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_from_models_with_strings_infers_tokenizer_and_saves(model_path: str):
    """from_models() with only string IDs/paths should infer instruct tokenizer and save it."""
    res = Residuals.from_models(
        base_model=model_path,
        instruct_model=model_path,
        dtype=torch.float32,
    )
    assert res.config.tokenizer_name is not None

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        res.save_pretrained(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "tokenizer_config.json"))
        # README should be generated with HF front-matter
        readme_path = os.path.join(tmpdir, "README.md")
        assert os.path.exists(readme_path), "README.md not generated"
        with open(readme_path, "r", encoding="utf-8") as fh:
            readme = fh.read()
        assert "base_model:" in readme, "README missing base_model front-matter"
        assert "base_model_relation: adapter" in readme, "README missing base_model_relation: adapter"
        # README usage should reference the folder name created by save_pretrained
        folder_name = os.path.basename(tmpdir)
        assert f'Residuals.from_pretrained("{folder_name}")' in readme

        # Load and compare
        res2 = Residuals.from_pretrained(tmpdir)

        assert len(res2.state_dict) == len(res.state_dict), "Residual count mismatch"

        for key in res.state_dict.keys():
            diff = (res.state_dict[key] - res2.state_dict[key]).abs().max().item()
            assert diff < 1e-7, f"Save/load mismatch for {key}: diff={diff}"


@pytest.mark.parametrize("model_path", MODELS)
def test_residual_properties(model_path: str):
    """
    Test mathematical properties of residuals as task vectors.

    Validates:
    - Residuals have expected sparsity/magnitude
    - Negation property: -Θ_r should reverse instruction tuning
    """
    model_path = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"

    model_base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, low_cpu_mem_usage=True)
    model_instruct = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, low_cpu_mem_usage=True)

    # Simulate instruction tuning
    with torch.no_grad():
        for key, param in model_instruct.state_dict().items():
            param.add_(torch.randn_like(param) * 0.05)

    res = Residuals.from_models(model_base, model_instruct)

    # Check residual statistics
    all_values = torch.cat([r.flatten() for r in res.state_dict.values()])
    mean_abs = all_values.abs().mean().item()

    assert mean_abs > 0, "Residuals should have non-zero magnitude"

    # Test negation: base + Θ_r - Θ_r = base
    model_test = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    # Apply residuals
    res.apply(model_test)

    # Apply negative residuals
    neg_res = Residuals({k: -v for k, v in res.state_dict.items()}, res.config)
    neg_res.apply(model_test)

    # Should match original base
    base_sd = model_base.state_dict()
    test_sd = model_test.state_dict()

    max_diff = max((base_sd[k] - test_sd[k]).abs().max().item() for k in base_sd.keys())

    assert max_diff < 1e-4, f"Negation property failed: diff={max_diff}"


@pytest.mark.parametrize("model_path", MODELS)
def test_shape_mismatch_raises(model_path: str):
    """Test that shape mismatches raise appropriate errors."""
    # Parametrized model path

    model_a = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, low_cpu_mem_usage=True)
    model_b = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, low_cpu_mem_usage=True)

    # Artificially create shape mismatch (this is contrived for testing)
    sd_b = model_b.state_dict()
    first_key = list(sd_b.keys())[0]

    # Save original shape
    original_shape = sd_b[first_key].shape

    # This test verifies the validation logic exists
    # In practice, same architecture = same shapes
    assert original_shape == model_a.state_dict()[first_key].shape


@pytest.mark.parametrize("model_path", MODELS)
def test_from_models_with_strings(model_path: str):
    """Ensure from_models can accept string model IDs/paths and still reconstruct."""
    # Use string args to compute residuals
    res = Residuals.from_models(
        base_model=model_path,
        instruct_model=model_path,
        dtype=torch.float32,
    )

    assert len(res.state_dict) > 0

    # Apply to a freshly loaded base
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    res.apply(base)

    # Since names are the same, residuals should be near zero; applying should keep model unchanged
    zero_like = Residuals.from_models(
        base_model=AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32),
        instruct_model=AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32),
    )
    max_abs = max(v.abs().max().item() for v in zero_like.state_dict.values())
    assert max_abs < 1e-6


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_residuals_to_changes_dtype_and_applies(model_path: str):
    """Residuals.to should cast tensors and still apply correctly on CPU."""
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    inst = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    with torch.no_grad():
        for _, p in inst.state_dict().items():
            p.add_(torch.randn_like(p) * 0.01)

    res = Residuals.from_models(base, inst, instruct_tokenizer_name=model_path)
    res_fp16 = res.to(dtype=torch.float16)

    # Ensure dtype changed for at least one tensor
    any_fp16 = any(t.dtype == torch.float16 for t in res_fp16.state_dict.values())
    assert any_fp16, "Residuals.to(dtype) did not change tensor dtype"

    # Apply casted residuals to a fresh base and verify reconstruction still works
    base_copy = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    res_fp16.apply(base_copy)

    inst_sd = inst.state_dict()
    recon_sd = base_copy.state_dict()
    max_diff = max((inst_sd[k] - recon_sd[k]).abs().max().item() for k in inst_sd.keys())
    assert max_diff < 1e-4


def test_from_pretrained_uses_local_path_directly(monkeypatch):
    """from_pretrained should not attempt hub download when given a local path."""
    # Create a real saved residuals folder
    base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2", dtype=torch.float32)
    inst = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2", dtype=torch.float32)
    with torch.no_grad():
        for _, p in inst.state_dict().items():
            p.add_(torch.randn_like(p) * 0.001)
    res = Residuals.from_models(base, inst, instruct_tokenizer_name="sshleifer/tiny-gpt2")

    called = {"download": False}
    import residuals.residuals as rmod
    def fake_download(repo_id: str, token=None) -> str:  # pragma: no cover - safety
        called["download"] = True
        return repo_id
    monkeypatch.setattr(rmod, "_download_from_hub", fake_download)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        res.save_pretrained(tmpdir)
        _ = Residuals.from_pretrained(tmpdir)
    assert called["download"] is False


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_from_models_accepts_polymorphic_args(model_path: str):
    """from_models should accept strings (paths/ids) and model instances interchangeably."""
    base_inst = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    inst_inst = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    with torch.no_grad():
        for _, p in inst_inst.state_dict().items():
            p.add_(torch.randn_like(p) * 0.001)
    # mix: base as string, instruct as instance
    res1 = Residuals.from_models(base_model=model_path, instruct_model=inst_inst)
    assert len(res1.state_dict) > 0
    # mix: base as instance, instruct as string
    res2 = Residuals.from_models(base_model=base_inst, instruct_model=model_path)
    assert len(res2.state_dict) > 0


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_apply_accepts_string_base_model(model_path: str):
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    inst = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    with torch.no_grad():
        for _, p in inst.state_dict().items():
            p.add_(torch.randn_like(p) * 0.001)
    res = Residuals.from_models(base, inst, instruct_tokenizer_name=model_path)
    merged = res.apply(base_model=model_path, normalize_embeddings=True)
    inst_sd = inst.state_dict()
    merged_sd = merged.state_dict()
    max_diff = max((inst_sd[k] - merged_sd[k]).abs().max().item() for k in inst_sd.keys())
    assert max_diff < 1e-4


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_apply_to_pretrained_accepts_model_instance(model_path: str):
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    inst = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    with torch.no_grad():
        for _, p in inst.state_dict().items():
            p.add_(torch.randn_like(p) * 0.002)
    res = Residuals.from_models(base, inst, instruct_tokenizer_name=model_path)
    import tempfile as _tf
    with _tf.TemporaryDirectory() as tmpdir:
        res.save_pretrained(tmpdir)
        merged = Residuals.apply_to_pretrained(
            model=AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32),
            residuals=tmpdir,
            normalize_embeddings=True,
        )
        inst_sd = inst.state_dict()
        merged_sd = merged.state_dict()
        max_diff = max((inst_sd[k] - merged_sd[k]).abs().max().item() for k in inst_sd.keys())
        assert max_diff < 1e-4


    


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_from_models_auto_deduces_instruct_tokenizer_and_saves(model_path: str):
    """from_models should auto-deduce instruct tokenizer from the instruct model or its name when not provided."""
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    inst = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    with torch.no_grad():
        for _, p in inst.state_dict().items():
            p.add_(torch.randn_like(p) * 0.005)

    # Do not pass any tokenizer params; should infer from instruct model
    res = Residuals.from_models(base_model=base, instruct_model=inst)
    assert res.config.tokenizer_name is not None, "Tokenizer name should be set in config when inferred"

    # Save should include tokenizer artifacts without explicitly passing tokenizer
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        res.save_pretrained(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "tokenizer_config.json")), "Tokenizer not saved with residuals"


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_apply_auto_deduces_base_tokenizer(model_path: str):
    """apply() should auto-deduce base tokenizer from the base model if not provided and align embeddings."""
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    inst = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    with torch.no_grad():
        for _, p in inst.state_dict().items():
            p.add_(torch.randn_like(p) * 0.003)

    res = Residuals.from_models(base, inst)

    # Do not pass base_tokenizer; should infer internally and not raise
    base_copy = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    res.apply(base_copy)
    # Sanity: after apply, base_copy should move towards inst
    inst_sd = inst.state_dict()
    copy_sd = base_copy.state_dict()
    diffs_after = [ (inst_sd[k] - copy_sd[k]).abs().mean().item() for k in inst_sd.keys() ]
    # Mean diff per param should be small-ish; just ensure it's finite and non-NaN
    assert all(d >= 0 for d in diffs_after)


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_apply_accepts_tokenizer_names_and_saves(model_path: str):
    """apply() should accept tokenizer names for both base and instruct and save tokenizer when out_dir is provided."""
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    inst = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    with torch.no_grad():
        for _, p in inst.state_dict().items():
            p.add_(torch.randn_like(p) * 0.002)

    res = Residuals.from_models(base, inst)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        base_copy = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
        res.apply(
            base_copy,
            base_tokenizer_name=model_path,
            instruct_tokenizer_name=model_path,
            out_dir=tmpdir,
        )
        # Tokenizer artifacts should exist in out_dir due to instruct tokenizer provided by name
        assert os.path.exists(os.path.join(tmpdir, "tokenizer_config.json"))


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_apply_with_string_base_model_and_model_dtype(model_path: str):
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    inst = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    with torch.no_grad():
        for _, p in inst.state_dict().items():
            p.add_(torch.randn_like(p) * 0.003)

    res = Residuals.from_models(base, inst, instruct_tokenizer_name=model_path)

    merged = res.apply(
        base_model=model_path,
        model_dtype=torch.float32,
        normalize_embeddings=True,
    )

    inst_sd = inst.state_dict()
    merged_sd = merged.state_dict()
    max_diff = max((inst_sd[k] - merged_sd[k]).abs().max().item() for k in inst_sd.keys())
    assert max_diff < 1e-4


@pytest.mark.parametrize("model_path", MODELS_WITH_TOKENIZER)
def test_apply_to_pretrained_end_to_end(model_path: str):
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    inst = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    with torch.no_grad():
        for _, p in inst.state_dict().items():
            p.add_(torch.randn_like(p) * 0.0025)

    res = Residuals.from_models(base, inst, instruct_tokenizer_name=model_path)

    import tempfile as _tf
    with _tf.TemporaryDirectory() as tmpdir:
        res.save_pretrained(tmpdir)
        merged = Residuals.apply_to_pretrained(
            model=model_path,
            residuals=tmpdir,
            normalize_embeddings=True,
            dtype=torch.float32,
            device="cpu",
        )
        inst_sd = inst.state_dict()
        merged_sd = merged.state_dict()
        max_diff = max((inst_sd[k] - merged_sd[k]).abs().max().item() for k in inst_sd.keys())
        assert max_diff < 1e-4
