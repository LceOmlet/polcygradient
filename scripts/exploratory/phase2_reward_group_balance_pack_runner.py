#!/usr/bin/env python
"""Run the trusted pack PPO runner with exploratory reward group balancing.

The only behavior change is installed at process start: frozen_h entries
annotated by phase2_reward_group_balance_probe.py get a reward-path-only
transition wrapper.  PPO, VecNorm, logging, and the pack runner itself are
otherwise delegated to ticl.analysis.phase2_gym_prior_pack_training_runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase2_reward_group_balance_runtime import (  # noqa: E402
    install_environment_prior_reward_group_balance_patch,
)


def main(argv: list[str] | None = None) -> None:
    install_environment_prior_reward_group_balance_patch()
    from ticl.analysis.phase2_gym_prior_pack_training_runner import main as pack_main  # noqa: E402

    pack_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
