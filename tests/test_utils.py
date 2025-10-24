from types import SimpleNamespace

from residuals.metadata import infer_model_name, infer_tokenizer_name
from residuals.readme import build_front_matter, build_readme
from residuals.config import ResidualsConfig

def test_infer_model_name_from_config():
    fake_model = SimpleNamespace(config=SimpleNamespace(name_or_path="hf-org/fake-model"))
    assert infer_model_name(fake_model) == "hf-org/fake-model"


def test_infer_model_name_fallback_private_attr():
    fake_model = SimpleNamespace(config=SimpleNamespace(_name_or_path="hf-org/fake-model"))
    assert infer_model_name(fake_model) == "hf-org/fake-model"


def test_infer_tokenizer_name():
    fake_tok = SimpleNamespace(name_or_path="hf-org/fake-tokenizer")
    assert infer_tokenizer_name(fake_tok) == "hf-org/fake-tokenizer"


def test_build_front_matter_contains_lineage():
    cfg = ResidualsConfig(
        residuals_version="1.0",
        format="pytorch",
        dtype="float32",
        param_count=1,
        shapes_hash="abc",
        parameter_names_hash="def",
        tokenizer_name="hf-org/fake-tokenizer",
        created_at="2025-01-01T00:00:00Z",
        library={"package": "residuals"},
        base_model_name="hf-org/base",
        instruct_model_name="hf-org/instruct",
        pipeline_tag="text-generation",
        license="mit",
        language="en",
        tags=["residuals"],
        base_model_relation="adapter",
    )
    fm = build_front_matter(cfg)
    assert "base_model: hf-org/base" in fm
    assert "base_model_relation: adapter" in fm
    assert "instruct_model: hf-org/instruct" in fm


def test_build_readme_includes_sections():
    cfg = ResidualsConfig(
        residuals_version="1.0",
        format="pytorch",
        dtype="float32",
        param_count=1,
        shapes_hash="abc",
        parameter_names_hash="def",
        tokenizer_name="hf-org/fake-tokenizer",
        created_at="2025-01-01T00:00:00Z",
        library={"package": "residuals"},
        base_model_name="hf-org/base",
        instruct_model_name="hf-org/instruct",
        pipeline_tag="text-generation",
        license="mit",
        language="en",
        tags=["residuals"],
        base_model_relation="adapter",
    )
    readme = build_readme(cfg)
    assert "# Instruction Residuals" in readme
    assert "## Usage" in readme
    assert "instruct" in readme
    assert "adapter" in readme
