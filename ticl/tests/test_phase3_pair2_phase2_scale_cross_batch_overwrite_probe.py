from ticl.analysis.phase3_pair2_phase2_scale_cross_batch_overwrite_probe import _vector_alignment


def test_vector_alignment_reports_negative_projection_for_overwrite_direction() -> None:
    ref = __import__("torch").tensor([1.0, 0.0], dtype=__import__("torch").float32)
    cand = __import__("torch").tensor([-0.25, 0.0], dtype=__import__("torch").float32)
    report = _vector_alignment(ref, cand)
    assert report["cosine"] == -1.0
    assert report["projection_on_reference"] == -0.25


def test_vector_alignment_reports_positive_projection_for_same_direction() -> None:
    ref = __import__("torch").tensor([2.0, 0.0], dtype=__import__("torch").float32)
    cand = __import__("torch").tensor([1.0, 0.0], dtype=__import__("torch").float32)
    report = _vector_alignment(ref, cand)
    assert report["cosine"] == 1.0
    assert report["projection_on_reference"] == 0.5
