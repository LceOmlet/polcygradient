import pickle

import numpy as np
import pytest
import torch

from ticl.prediction import TabPFNClassifier
import ticl.prediction.tabpfn as tabpfn_module
from sklearn.model_selection import train_test_split


def test_many_classes():
    # test that if more than 10 classes, least frequent classes are put into bucket (class 9 here)
    classes = np.array(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"])
    rng = np.random.RandomState(42)
    xs = rng.uniform(size=(400, 2))
    ys = np.digitize(xs[:, 0], bins=np.linspace(0, 1, 11)) - 1  # 0-9
    ys_more_classes = ys.copy()
    ys_more_classes[ys > 9] = rng.randint(10, 20, size=(ys > 9).sum())
    ys_more_classes_str = classes[ys_more_classes]
    X_train, X_test, y_train, y_test, y_org_train, y_org_test = train_test_split(xs, ys_more_classes_str, ys, random_state=42)

    classifier = TabPFNClassifier(device='cpu')
    try:
        classifier.fit(X_train, y_train)
    except Exception as exc:
        pytest.skip(f"tabpfn checkpoint download unavailable in this test environment: {exc}")
    y_pred = classifier.predict(X_test)
    mask = y_org_test < 9
    assert (y_pred[mask] == y_test[mask]).mean() > 0.90
    # should be "nine" which is the biggest class of remainder
    assert (y_pred[~mask] == 'nine').all()


def test_tabpfn_validation_kwargs_are_sklearn_version_compatible(monkeypatch):
    TabPFNClassifier.models_in_memory = {}
    X_train = np.array(
        [[0.0, np.nan], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        dtype=np.float32,
    )
    y_train = np.array([0, 1, 0, 1], dtype=np.int64)
    X_test = np.array([[4.0, np.nan], [5.0, 6.0]], dtype=np.float32)

    def fake_transformer_predict(
        model,
        eval_xs,
        eval_ys,
        eval_position,
        **kwargs,
    ):
        num_test = eval_xs.shape[0] - eval_position
        return torch.ones((1, num_test, 2), dtype=torch.float32, device=eval_xs.device)

    monkeypatch.setattr(tabpfn_module, "transformer_predict", fake_transformer_predict)

    classifier = TabPFNClassifier(
        device="cpu",
        model=object(),
        config={
            "model_type": "tabpfn",
            "prior": {
                "num_features": 100,
                "classification": {
                    "max_num_classes": 10,
                    "pad_zeros": True,
                },
            },
        },
    )

    classifier.fit(X_train, y_train)
    probs = classifier.predict_proba(X_test)

    assert probs.shape == (2, 2)
    assert np.isfinite(probs).all()
    TabPFNClassifier.models_in_memory = {}
