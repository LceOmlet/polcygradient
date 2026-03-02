import numpy as np

from ticl.models.encoders import Linear
from ticl.models.tabpfn import TabPFN
from ticl.rl_validation import _score_candidate_actions


def _build_small_causal_model():
    model = TabPFN(
        n_out=1,
        n_features=12,
        emsize=32,
        nhead=1,
        nhid_factor=2,
        nlayers=1,
        y_encoder_layer=Linear(1, emsize=32),
        classification_task=False,
        y_encoder="linear",
        x_encoder_type="single",
        single_eval_causal=True,
    )
    model.eval()
    return model


def test_score_candidate_actions_bootstrap_step_no_prefix_no_crash():
    model = _build_small_causal_model()
    candidates = np.array(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.3, 0.4],
        ],
        dtype=np.float32,
    )
    scores = _score_candidate_actions(
        model=model,
        device="cpu",
        x_hist=[],
        y_hist=[],
        obs=np.zeros((4,), dtype=np.float32),
        prev_reward=0.0,
        action_candidates=candidates,
        obs_slot_dim=8,
        action_slot_dim=2,
        num_features=12,
    )

    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert np.allclose(scores, 0.0)
