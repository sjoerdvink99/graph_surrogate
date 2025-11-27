import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def output_dir(tmp_path):
    yield tmp_path / "output"


@pytest.fixture
def streaming_dir(tmp_path):
    yield tmp_path / "streaming"


def run_cmd(args: list[str]) -> None:
    cmd = ["uv", "run", "python", "-m"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
    assert result.returncode == 0


def test_download_dataset():
    run_cmd(["datasets.registry", "download", "deezer"])


def test_train(output_dir):
    run_cmd([
        "training.train",
        "--dataset", "deezer",
        "--output-dir", str(output_dir),
        "--num-train", "1000",
        "--num-val", "200",
        "--num-test", "200",
        "--epochs", "5",
        "--batch-size", "64",
        "--patience", "3",
    ])
    assert (output_dir / "model.pt").exists()


def test_evaluate(output_dir):
    run_cmd([
        "training.train",
        "--dataset", "deezer",
        "--output-dir", str(output_dir),
        "--num-train", "500",
        "--num-val", "100",
        "--num-test", "100",
        "--epochs", "2",
        "--batch-size", "64",
    ])
    run_cmd([
        "scripts.evaluate",
        "--model-dir", str(output_dir),
        "--num-samples", "100",
        "--skip-latency",
    ])
    assert (output_dir / "evaluation_results.json").exists()


def test_streaming(streaming_dir):
    run_cmd([
        "training.train",
        "--dataset", "deezer",
        "--output-dir", str(streaming_dir),
        "--num-train", "500",
        "--num-val", "100",
        "--num-test", "100",
        "--epochs", "2",
        "--batch-size", "32",
        "--streaming",
    ])
    assert (streaming_dir / "model.pt").exists()
