"""CLI imports must not emit Streamlit cache runtime warnings."""

import os
import subprocess
import sys
from pathlib import Path


def test_pipeline_import_emits_no_streamlit_runtime_warning() -> None:
    """Fresh interpreter importing pipeline stays quiet about Streamlit cache."""
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from happy_housing.pipeline import parse_pipeline_cli_args; "
            "parse_pipeline_cli_args([])",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(root),
        env=env,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, combined
    assert "No runtime found" not in combined
    assert "MemoryCacheStorageManager" not in combined
