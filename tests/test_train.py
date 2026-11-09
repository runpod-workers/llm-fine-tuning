import asyncio
import stat
import textwrap

import pytest

from train import train


@pytest.fixture
def fake_axolotl(tmp_path, monkeypatch):
    """Put an `axolotl` on PATH whose exit code per subcommand comes from env."""
    script = tmp_path / "axolotl"
    script.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            echo "axolotl $1 running"
            var="FAKE_EXIT_$1"
            eval code=\\${$var:-0}
            exit $code
            """
        )
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}:{__import__('os').environ['PATH']}")
    return monkeypatch


def collect(config_path="cfg.yaml", **kwargs):
    async def run():
        return [line async for line in train(config_path, **kwargs)]

    return asyncio.run(run())


def test_streams_preprocess_and_train_output(fake_axolotl):
    lines = collect()

    assert "Preprocessing: axolotl preprocess running" in lines
    assert "Training: axolotl train running" in lines


def test_raises_when_preprocess_fails_and_skips_training(fake_axolotl):
    fake_axolotl.setenv("FAKE_EXIT_preprocess", "3")
    seen = []

    async def run():
        async for line in train("cfg.yaml"):
            seen.append(line)

    with pytest.raises(RuntimeError, match="preprocess exited with code 3"):
        asyncio.run(run())
    assert not any(line.startswith("Training:") for line in seen)


def test_raises_when_training_fails(fake_axolotl):
    fake_axolotl.setenv("FAKE_EXIT_train", "1")

    with pytest.raises(RuntimeError, match="train exited with code 1"):
        collect()
