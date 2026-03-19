from ticl.model_configs import get_prior_config
import lightning as L
import torch
import pytest
import numpy as np

from ticl.priors import ClassificationAdapterPrior, MLPPrior
from ticl.priors.classification_adapter import ClassificationAdapter


@pytest.mark.parametrize("num_features", [11, 51])
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.parametrize("n_samples", [128, 900])
def test_classification_prior_no_sampling(batch_size, num_features, n_samples, n_classes):
    # test the mlp prior
    L.seed_everything(43)
    config = get_prior_config()
    config['prior']['classification']['num_features_used'] = num_features  # always using all features in this test
    config['prior']['classification']['num_classes'] = n_classes
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

    prior = ClassificationAdapterPrior(MLPPrior(config['prior']['mlp']), **config['prior']['classification'])

    x, y, y_, info = prior.get_batch(batch_size=batch_size, num_features=num_features, n_samples=n_samples, device='cpu')
    assert x.shape == (n_samples, batch_size, num_features)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)
    # because of the strange sampling, we an have less than n_classes many classes.
    assert y_.max() < n_classes
    assert y_.min() == 0
    if n_samples == 128 and batch_size == 4 and num_features == 11 and n_classes == 2:
        assert float(x[0, 0, 0]) == 1.4561537504196167
        assert float(y[0, 0]) == 1.0


def test_classification_adapter_with_sampling():
    batch_size = 16
    num_features = 100
    n_samples = 900
    # test the mlp prior
    L.seed_everything(42)
    config = get_prior_config()
    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), config=config['prior']['classification'])
    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features}
    x, y, y_, info = adapter(batch_size=batch_size, **args)
    assert x.shape == (n_samples, batch_size, num_features)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)
    # Numerical snapshots are brittle across torch/numpy RNG backend changes.
    # Validate deterministic replay under identical seed instead.
    L.seed_everything(42)
    config_replay = get_prior_config()
    adapter_replay = ClassificationAdapter(
        MLPPrior(config_replay['prior']['mlp']),
        config=config_replay['prior']['classification'],
    )
    x_replay, y_replay, y_replay_, _ = adapter_replay(batch_size=batch_size, **args)

    assert torch.allclose(x, x_replay)
    assert torch.allclose(y, y_replay)
    assert torch.equal(y_, y_replay_)


def test_classification_adapter_curriculum():
    batch_size = 16
    num_features = 100
    n_samples = 900
    # test the mlp prior
    L.seed_everything(42)
    config = get_prior_config()
    classification_config = config['prior']['classification']
    classification_config['feature_curriculum'] = True
    classification_config['pad_zeros'] = False

    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), config=classification_config)
    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features, 'epoch': 0}
    x, y, y_, info = adapter(batch_size=batch_size, **args)
    n_epoch0 = int(x.shape[-1])
    assert x.shape == (n_samples, batch_size, n_epoch0)
    assert n_epoch0 == 1
    args['epoch'] = 1
    x, y, y_, info = adapter(batch_size=batch_size, **args)
    n_epoch1 = int(x.shape[-1])
    assert x.shape == (n_samples, batch_size, n_epoch1)
    assert n_epoch1 >= n_epoch0
    args['epoch'] = 100
    x, y, y_, info = adapter(batch_size=batch_size, **args)
    n_epoch100 = int(x.shape[-1])
    assert x.shape == (n_samples, batch_size, n_epoch100)
    assert n_epoch100 > n_epoch1
    assert n_epoch100 <= num_features


def test_classification_adapter_double_sampler():
    batch_size = 16
    num_features = 100
    n_samples = 900
    # test the mlp prior
    L.seed_everything(42)
    config = get_prior_config()
    classification_config = config['prior']['classification']
    classification_config['num_features_sampler'] = 'double_sample'
    classification_config['pad_zeros'] = False

    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), config=classification_config)
    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features, 'epoch': 0}
    num_features = np.array([adapter(batch_size=batch_size, **args)[0].shape[-1] for i in range(10)])
    assert num_features.min() >= 1
    assert num_features.max() <= int(args["num_features"])
    assert len(np.unique(num_features)) > 1
    assert (num_features < 20).any()
    assert (num_features > 30).any()


def test_classification_adapter_with_sampling_no_padding():
    batch_size = 16
    num_features = 100
    n_samples = 900
    # test the mlp prior
    L.seed_everything(42)
    config = get_prior_config()
    prior_config = config['prior']['classification']
    prior_config['pad_zeros'] = False
    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), config=prior_config)

    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features}
    x, y, y_, info = adapter(batch_size=batch_size, **args)
    assert x.shape == (n_samples, batch_size, 72)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)

    L.seed_everything(42)
    replay_config = get_prior_config()
    replay_prior_config = replay_config['prior']['classification']
    replay_prior_config['pad_zeros'] = False
    adapter_replay = ClassificationAdapter(MLPPrior(replay_config['prior']['mlp']), config=replay_prior_config)
    x_replay, y_replay, y_replay_, _ = adapter_replay(batch_size=batch_size, **args)

    assert torch.allclose(x, x_replay)
    assert torch.allclose(y, y_replay)
    assert torch.equal(y_, y_replay_)


def test_classification_adapter_nan():
    batch_size = 16
    num_features = 100
    n_samples = 900
    # test the mlp prior
    L.seed_everything(12)
    config = get_prior_config()
    prior_config = config['prior']['classification']
    prior_config['pad_zeros'] = False
    prior_config['nan_prob_no_reason'] = 0.99
    prior_config['nan_prob_a_reason'] = 0
    prior_config['set_value_to_nan'] = 1.0

    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), config=prior_config)

    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features}
    nan_fracs = []
    for _ in range(16):
        x, y, y_, _ = adapter(batch_size=batch_size, **args)
        assert y.shape == (n_samples, batch_size)
        nan_fracs.append(float(x.isnan().float().mean().item()))
    assert max(nan_fracs) > 0.05

    prior_config['nan_prob_no_reason'] = 0
    prior_config['nan_prob_a_reason'] = 0.99
    adapter = ClassificationAdapter(MLPPrior(config['prior']['mlp']), config=prior_config)

    args = {'device': 'cpu', 'n_samples': n_samples, 'num_features': num_features}
    x, y, y_, _ = adapter(batch_size=batch_size, **args)
    assert y.shape == (n_samples, batch_size)
    assert y_.shape == (n_samples, batch_size)
    assert x.isnan().float().mean() > 0.45
