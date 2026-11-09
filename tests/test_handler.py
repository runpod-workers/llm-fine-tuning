import asyncio
import importlib
import os

import pytest
import yaml


@pytest.fixture
def handler(tmp_path, monkeypatch):
    import runpod.serverless

    monkeypatch.setattr(runpod.serverless, "start", lambda config: None)
    monkeypatch.setenv("BASE_VOLUME", str(tmp_path / "volume"))
    import handler as module

    module = importlib.reload(module)
    monkeypatch.setattr(module, "CONFIG_PATH", str(tmp_path / "config.yaml"))
    return module


def fake_train(lines=("Training: ok",), error=None):
    async def train(config_path):
        for line in lines:
            yield line
        if error:
            raise error

    return train


def job(credentials=None):
    data = {"id": "job-1", "input": {"run_id": "run-1", "args": {"base_model": "tiny"}}}
    if credentials is not None:
        data["input"]["credentials"] = credentials
    return data


def run(handler, data):
    return asyncio.run(handler.handler(data))


def test_returns_run_id_and_output_dir(handler, monkeypatch, tmp_path):
    monkeypatch.setattr(handler, "train", fake_train())

    result = run(handler, job())

    assert result == {
        "run_id": "run-1",
        "output_dir": str(tmp_path / "volume" / "fine-tuning" / "run-1"),
    }


def test_writes_axolotl_config_with_run_metadata(handler, monkeypatch):
    monkeypatch.setattr(handler, "train", fake_train())

    run(handler, job())

    with open(handler.CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    assert config["base_model"] == "tiny"
    assert config["run_name"] == "run-1"
    assert config["runpod_job_id"] == "job-1"


@pytest.fixture
def hf_home(handler, tmp_path, monkeypatch):
    """Isolated HF cache dir plus a guard that fails on any persistent login."""
    import huggingface_hub
    import huggingface_hub._login as hf_login

    home = tmp_path / "hf_home"
    monkeypatch.setenv("HF_HOME", str(home))

    def forbidden(*args, **kwargs):
        raise AssertionError("persistent huggingface_hub login must not be used")

    monkeypatch.setattr(hf_login, "login", forbidden)
    monkeypatch.setattr(huggingface_hub, "login", forbidden)
    monkeypatch.setattr(handler, "login", forbidden, raising=False)
    return home


def capture_env(seen):
    async def train(config_path):
        seen.append({k: os.environ.get(k) for k in ("HF_TOKEN", "WANDB_API_KEY")})
        yield "Training: ok"

    return train


def test_runs_without_credentials(handler, monkeypatch, hf_home):
    seen = []
    monkeypatch.setattr(handler, "train", capture_env(seen))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    run(handler, job())

    assert seen == [{"HF_TOKEN": None, "WANDB_API_KEY": None}]


def test_sets_job_credentials_in_env_without_persisting_login(handler, monkeypatch, hf_home):
    seen = []
    monkeypatch.setattr(handler, "train", capture_env(seen))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    run(handler, job({"hf_token": "hf_x", "wandb_api_key": "wb_x"}))

    assert seen == [{"HF_TOKEN": "hf_x", "WANDB_API_KEY": "wb_x"}]
    assert not (hf_home / "token").exists()
    assert "HF_TOKEN" not in os.environ
    assert "WANDB_API_KEY" not in os.environ


def test_credential_free_job_after_authenticated_job_sees_no_token(handler, monkeypatch, hf_home):
    seen = []
    monkeypatch.setattr(handler, "train", capture_env(seen))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    run(handler, job({"hf_token": "hf_first", "wandb_api_key": "wb_first"}))
    run(handler, job())

    assert seen[1] == {"HF_TOKEN": None, "WANDB_API_KEY": None}
    assert "HF_TOKEN" not in os.environ
    assert not (hf_home / "token").exists()


def test_restores_worker_env_values_overridden_by_job(handler, monkeypatch, hf_home):
    seen = []
    monkeypatch.setattr(handler, "train", capture_env(seen))
    monkeypatch.setenv("HF_TOKEN", "worker_hf")
    monkeypatch.setenv("WANDB_API_KEY", "worker_wb")

    run(handler, job({"hf_token": "job_hf", "wandb_api_key": "job_wb"}))

    assert seen == [{"HF_TOKEN": "job_hf", "WANDB_API_KEY": "job_wb"}]
    assert os.environ["HF_TOKEN"] == "worker_hf"
    assert os.environ["WANDB_API_KEY"] == "worker_wb"


def test_keeps_worker_env_values_the_job_does_not_override(handler, monkeypatch, hf_home):
    monkeypatch.setattr(handler, "train", fake_train())
    monkeypatch.setenv("HF_TOKEN", "worker_hf")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    run(handler, job({"wandb_api_key": "job_wb"}))

    assert os.environ["HF_TOKEN"] == "worker_hf"
    assert "WANDB_API_KEY" not in os.environ


def test_training_failure_propagates_and_clears_absent_credentials(handler, monkeypatch, hf_home):
    monkeypatch.setattr(handler, "train", fake_train(error=RuntimeError("axolotl train exited with code 1")))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="exited with code 1"):
        run(handler, job({"hf_token": "hf_x", "wandb_api_key": "wb_x"}))

    assert "HF_TOKEN" not in os.environ
    assert "WANDB_API_KEY" not in os.environ


def test_training_failure_restores_worker_env_values(handler, monkeypatch, hf_home):
    monkeypatch.setattr(handler, "train", fake_train(error=RuntimeError("axolotl train exited with code 1")))
    monkeypatch.setenv("HF_TOKEN", "worker_hf")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="exited with code 1"):
        run(handler, job({"hf_token": "job_hf", "wandb_api_key": "job_wb"}))

    assert os.environ["HF_TOKEN"] == "worker_hf"
    assert "WANDB_API_KEY" not in os.environ
