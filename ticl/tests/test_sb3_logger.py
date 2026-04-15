from io import StringIO
import importlib.util
from pathlib import Path


LOGGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "stable_baselines3"
    / "common"
    / "logger.py"
)
LOGGER_SPEC = importlib.util.spec_from_file_location("sb3_logger_module_for_test", LOGGER_PATH)
assert LOGGER_SPEC is not None and LOGGER_SPEC.loader is not None
LOGGER_MODULE = importlib.util.module_from_spec(LOGGER_SPEC)
LOGGER_SPEC.loader.exec_module(LOGGER_MODULE)
HumanOutputFormat = LOGGER_MODULE.HumanOutputFormat


def test_human_output_format_handles_truncation_collision_by_extending_key():
    stream = StringIO()
    fmt = HumanOutputFormat(stream, max_length=36)
    key_values = {
        "rollout/full_reward_terminal_bonus_return_mean": 1.0,
        "rollout/full_reward_terminal_bonus_return_std": 2.0,
    }
    key_excluded = {key: ("json", "csv", "tensorboard") for key in key_values}

    fmt.write(key_values, key_excluded, step=0)

    output = stream.getvalue()
    metric_lines = [line for line in output.splitlines() if "full_reward_terminal" in line]
    assert len(metric_lines) == 2
    assert len(set(metric_lines)) == 2
