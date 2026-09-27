"""Machine and code provenance for training runs (protocol addendum 2026-09-28 §2).

Results are only bit-identical on the same machine and code, so every run
records where and on what it ran. Stored in each run's `run_config.json`; the
MAPPO resume check compares it too, so a resume on another machine or on
changed code is refused rather than silently continuing.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

from udt.common.config import REPO_ROOT


def _cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or "unknown"


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def machine_info() -> dict[str, Any]:
    """Platform, CPU, library versions and code commit. No hostname/user."""
    import numpy
    import torch

    try:
        import stable_baselines3

        sb3 = stable_baselines3.__version__
    except ImportError:  # pragma: no cover - SB3 is a project dependency
        sb3 = "not installed"
    status = _git("status", "--porcelain", "--", "src", "scripts", "configs", "data")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "stable_baselines3": sb3,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
    }
