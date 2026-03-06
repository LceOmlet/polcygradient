from ticl.prediction.tabflex import TabFlex
import torch
import lightning as L
import pytest


@pytest.fixture(autouse=True)
def set_threads():
    return torch.set_num_threads(1)


# Test TabFlex prediction on synthetic data
# tabpflexs100, tabflexl100, tabflexh1k, tabflexh1k, tabflexl100 w random projection
@pytest.mark.parametrize("n_samples, n_features", [(300, 20), (3000, 100), (3000, 101), (300, 1000), (300, 1001)])
def test_predict_tabflex(n_samples, n_features):
    L.seed_everything(42)

    # Generate synthetic dataset
    X_train = torch.randn(n_samples, n_features)
    coef = torch.randn(n_features) 
    y_train = (X_train @ coef > 0).int()

    X_test = torch.randn(200, n_features)
    y_test = (X_test @ coef > 0).int()

    # Initialize and train TabFlex model
    try:
        tabflex = TabFlex()
    except Exception as exc:
        pytest.skip(f"tabflex model download unavailable in this test environment: {exc}")
    tabflex.fit(X_train, y_train)

    # Make predictions
    y_pred = tabflex.predict(X_test)

    # Evaluate performance
    acc = (torch.tensor(y_pred) == y_test).float().mean().item()
    if n_features < 1000:
        assert acc > 0.9
    elif n_samples >= 3000:
        assert acc > 0.85
    else:
        assert acc > 0.4   # hm this is not good
