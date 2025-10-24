import os
import tempfile

import pytest
import torch
from transformers import AutoTokenizer

from residuals import Residuals
from residuals.config import ResidualsConfig


def _make_dummy_residuals(tokenizer_name: str = "sshleifer/tiny-gpt2") -> Residuals:
    # Minimal state and config
    state = {"dummy.weight": torch.zeros(1)}
    cfg = ResidualsConfig(
        residuals_version="1.0",
        format="pytorch",
        dtype="float32",
        param_count=1,
        shapes_hash="shash",
        parameter_names_hash="nhash",
        tokenizer_name=tokenizer_name,
        created_at="2025-01-01T00:00:00Z",
        library={"package": "residuals"},
        base_model_name="hf-org/base",
        instruct_model_name="hf-org/instruct",
    )
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    return Residuals(state, cfg, instruct_tokenizer=tok)


@pytest.mark.parametrize("tokenizer_name", ["sshleifer/tiny-gpt2"])
def test_from_pretrained_hub_downloads_and_loads(monkeypatch, tokenizer_name: str):
    # Prepare a real saved residuals folder
    res = _make_dummy_residuals(tokenizer_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        res.save_pretrained(tmpdir)

        # Monkeypatch downloader to return our tmpdir
        import residuals.residuals as rmod

        def fake_download(repo_id: str, token=None) -> str:
            return tmpdir

        monkeypatch.setattr(rmod, "_download_from_hub", fake_download)

        # Call from_pretrained with a hub-like path
        res2 = Residuals.from_pretrained("username/model-residuals")

        assert len(res2.state_dict) == len(res.state_dict)
        assert res2.config.base_model_name == res.config.base_model_name


def test_push_to_hub_invokes_api(monkeypatch):
    res = _make_dummy_residuals()

    created = {}
    uploaded = {}

    class FakeApi:
        def create_repo(self, repo_id, private=False, exist_ok=True, token=None):
            created["repo_id"] = repo_id
            created["private"] = private
            created["exist_ok"] = exist_ok
            created["token"] = token

    def fake_upload_folder(folder_path, repo_id, token=None, path_in_repo=""):
        uploaded["folder_path"] = folder_path
        uploaded["repo_id"] = repo_id
        uploaded["token"] = token
        uploaded["path_in_repo"] = path_in_repo
        # sanity: required files exist
        assert os.path.exists(os.path.join(folder_path, "model.safetensors"))
        assert os.path.exists(os.path.join(folder_path, "config.json"))
        assert os.path.exists(os.path.join(folder_path, "README.md"))

    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    import residuals.residuals as rmod
    monkeypatch.setitem(rmod.__dict__, "HfApi", FakeApi)
    monkeypatch.setitem(rmod.__dict__, "upload_folder", fake_upload_folder)

    url = res.push_to_hub("username/model-residuals", private=True, token="hf_fake")

    assert url.endswith("username/model-residuals")
    assert created["repo_id"] == "username/model-residuals"
    assert created["private"] is True
    assert uploaded["repo_id"] == "username/model-residuals"
