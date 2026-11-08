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


def test_runs_without_credentials(handler, monkeypatch):
    logins = []
    monkeypatch.setattr(handler, "train", fake_train())
    monkeypatch.setattr(handler, "login", lambda token: logins.append(token))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    run(handler, job())

    assert logins == []


def test_logs_in_and_clears_credentials(handler, monkeypatch):
    logins = []
    seen_env = {}

    async def train(config_path):
        seen_env.update(HF_TOKEN=os.environ.get("HF_TOKEN"), WANDB=os.environ.get("WANDB_API_KEY"))
        yield "Training: ok"

    monkeypatch.setattr(handler, "train", train)
    monkeypatch.setattr(handler, "login", lambda token: logins.append(token))

    run(handler, job({"hf_token": "hf_x", "wandb_api_key": "wb_x"}))

    assert logins == ["hf_x"]
    assert seen_env == {"HF_TOKEN": "hf_x", "WANDB": "wb_x"}
    assert "HF_TOKEN" not in os.environ
    assert "WANDB_API_KEY" not in os.environ


def test_training_failure_propagates_and_clears_credentials(handler, monkeypatch):
    monkeypatch.setattr(handler, "train", fake_train(error=RuntimeError("axolotl train exited with code 1")))
    monkeypatch.setattr(handler, "login", lambda token: None)

    with pytest.raises(RuntimeError, match="exited with code 1"):
        run(handler, job({"hf_token": "hf_x", "wandb_api_key": "wb_x"}))

    assert "HF_TOKEN" not in os.environ
    assert "WANDB_API_KEY" not in os.environ
