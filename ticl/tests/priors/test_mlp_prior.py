from ticl.priors import MLPPrior
from ticl.model_configs import get_prior_config
import lightning as L
import torch
import pytest


@pytest.mark.parametrize("num_features", [11, 51])
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("n_samples", [128, 900])
def test_mlp_prior(batch_size, num_features, n_samples):
    # test the mlp prior
    L.seed_everything(42)
    config = get_prior_config()
    prior = MLPPrior(config['prior']['mlp'])

    x, y, y_ = prior.get_batch(batch_size=batch_size, num_features=num_features, n_samples=n_samples, device='cpu')
    assert x.shape == (n_samples, batch_size, num_features)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)
    if n_samples == 128 and batch_size == 4 and num_features == 11:
        L.seed_everything(42)
        config_replay = get_prior_config()
        prior_replay = MLPPrior(config_replay['prior']['mlp'])
        x_replay, y_replay, y_replay_ = prior_replay.get_batch(
            batch_size=batch_size,
            num_features=num_features,
            n_samples=n_samples,
            device='cpu',
        )
        assert torch.allclose(x, x_replay)
        assert torch.allclose(y, y_replay)
        assert torch.equal(y_, y_replay_)


def test_mlp_prior_no_sampling(batch_size=4, num_features=11, n_samples=128):
    # test the mlp prior
    L.seed_everything(42)
    config = get_prior_config()
    # replace distributions with some values for this test
    hyperparameters = {
        'prior_mlp_activations': torch.nn.ReLU,
        'is_causal': False,
        'num_causes': 3,  # actually ignored because is_causal is False
        'prior_mlp_hidden_dim': 128,
        'num_layers': 3,
        'noise_std': 0.1,
        'y_is_effect': True,
        'pre_sample_weights': False,
        'prior_mlp_dropout_prob': 0.1,
        'block_wise_dropout': True,
        'init_std': 0.1,
        'sort_features': False,
        'in_clique': False,
    }
    config['prior']['mlp'].update(hyperparameters)
    prior = MLPPrior(config['prior']['mlp'])

    x, y, y_ = prior.get_batch(batch_size=batch_size, num_features=num_features, n_samples=n_samples, device='cpu')
    assert x.shape == (n_samples, batch_size, num_features)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)
    L.seed_everything(42)
    replay_cfg = get_prior_config()
    replay_cfg['prior']['mlp'].update(hyperparameters)
    replay_prior = MLPPrior(replay_cfg['prior']['mlp'])
    x_replay, y_replay, y_replay_ = replay_prior.get_batch(
        batch_size=batch_size,
        num_features=num_features,
        n_samples=n_samples,
        device='cpu',
    )
    assert torch.allclose(x, x_replay)
    assert torch.allclose(y, y_replay)
    assert torch.equal(y_, y_replay_)
