from pathlib import Path

from scripts.exploratory.phase2_validation_mode_witness_probe import build_witness_report


def test_validation_mode_witness_selects_terminal_survival_without_mutation(tmp_path: Path):
    log_path = tmp_path / "validation.log"
    log_path.write_text(
        "\n".join(
            [
                "Epoch 1 return_Ant-v5_len_mean 1000",
                "Epoch 1 return_Ant-v5_return_mean 990",
                "Epoch 1 return_Ant-v5_reward_ctrl_mean -5",
                "Epoch 1 return_Ant-v5_reward_forward_mean 2",
                "Epoch 1 return_Ant-v5_reward_survive_mean 1000",
                "Epoch 2 return_Ant-v5_len_mean 1000",
                "Epoch 2 return_Ant-v5_return_mean 930",
                "Epoch 2 return_Ant-v5_reward_ctrl_mean -70",
                "Epoch 2 return_Ant-v5_reward_forward_mean 1",
                "Epoch 2 return_Ant-v5_reward_survive_mean 1000",
                "Epoch 3 return_Ant-v5_len_mean 1000",
                "Epoch 3 return_Ant-v5_return_mean 950",
                "Epoch 3 return_Ant-v5_reward_ctrl_mean -50",
                "Epoch 3 return_Ant-v5_reward_forward_mean 0",
                "Epoch 3 return_Ant-v5_reward_survive_mean 1000",
                "Epoch 1 return_InvertedPendulum-v5_len_mean 10",
                "Epoch 1 return_InvertedPendulum-v5_return_mean 9",
                "Epoch 1 return_InvertedPendulum-v5_reward_survive_mean 9",
                "Epoch 2 return_InvertedPendulum-v5_len_mean 20",
                "Epoch 2 return_InvertedPendulum-v5_return_mean 19",
                "Epoch 2 return_InvertedPendulum-v5_reward_survive_mean 19",
                "Epoch 3 return_InvertedPendulum-v5_len_mean 30",
                "Epoch 3 return_InvertedPendulum-v5_return_mean 29",
                "Epoch 3 return_InvertedPendulum-v5_reward_survive_mean 29",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_witness_report(validation_log=log_path)

    assert report["exploratory_only"] is True
    assert report["mutates_exact_scm"] is False
    assert report["mutates_ppo"] is False
    assert report["selected_next_witness"] == "terminal_survival_action_sensitive_counterfactual"
    assert report["sufficient_to_change_milestone"] is False
    assert report["inverted_terminal_survival"]["InvertedPendulum-v5"]["necessary_signature_pass"] is True
    assert report["ant_locomotion_energy"]["terminal_not_primary_signature_pass"] is True
