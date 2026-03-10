import numpy as np
import openml

from ticl import datasets as datasets_module


def test_get_openml_classification_uses_processed_cache_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(openml.config, "get_cache_directory", lambda: str(tmp_path))

    datasets_module._save_processed_openml_dataset(
        did=123,
        classification=True,
        multiclass=True,
        shuffled=True,
        X=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        y=np.array([0, 1, 0], dtype=np.int64),
        categorical_feats=[1],
        attribute_names=["f0", "f1"],
        name="cached-dataset",
    )

    def _fail(*args, **kwargs):
        raise AssertionError("network-backed get_dataset should not be called when processed cache exists")

    monkeypatch.setattr(openml.datasets, "get_dataset", _fail)

    X, y, categorical_feats, attribute_names = datasets_module.get_openml_classification(
        did=123,
        max_samples=2,
        multiclass=True,
        shuffled=True,
    )

    assert X.shape == (2, 2)
    assert y.shape == (2,)
    assert categorical_feats == [1]
    assert attribute_names == ["f0", "f1"]


def test_load_openml_list_uses_processed_cache_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(openml.config, "get_cache_directory", lambda: str(tmp_path))

    datasets_module._save_processed_openml_dataset(
        did=456,
        classification=True,
        multiclass=True,
        shuffled=True,
        X=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        y=np.array([0, 1, 0], dtype=np.int64),
        categorical_feats=[0],
        attribute_names=["a", "b"],
        name="cached-list-dataset",
    )

    def _fail(*args, **kwargs):
        raise AssertionError("network-backed OpenML access should not be called when processed cache exists")

    monkeypatch.setattr(openml.datasets, "get_dataset", _fail)
    monkeypatch.setattr(openml.datasets, "list_datasets", _fail)

    datasets, datalist = datasets_module.load_openml_list(
        [456],
        classification=True,
        filter_for_nan=False,
        num_feats=100,
        min_samples=1,
        max_samples=2,
        multiclass=True,
        max_num_classes=10,
        shuffled=True,
        return_capped=True,
    )

    assert len(datasets) == 1
    assert datasets[0][0] == "cached-list-dataset"
    assert datasets[0][1].shape == (2, 2)
    assert "NumberOfClasses" in datalist.columns
    assert datalist.iloc[0]["name"] == "cached-list-dataset"
