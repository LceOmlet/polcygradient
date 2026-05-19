__all__ = ["TabPFNClassifier"]


def __getattr__(name):
    if name == "TabPFNClassifier":
        from ticl.prediction.tabpfn import TabPFNClassifier

        return TabPFNClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
