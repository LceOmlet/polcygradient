from ticl.analysis.phase3_pair2_phase2_scale_local_step_direction_probe import _vector_alignment


def test_vector_alignment_positive_projection() -> None:
    torch = __import__("torch")
    ref = torch.tensor([1.0, 0.0], dtype=torch.float32)
    cand = torch.tensor([2.0, 0.0], dtype=torch.float32)
    out = _vector_alignment(ref, cand)
    assert out["cosine"] == 1.0
    assert out["projection_on_reference"] == 2.0


def test_vector_alignment_negative_projection() -> None:
    torch = __import__("torch")
    ref = torch.tensor([1.0, 0.0], dtype=torch.float32)
    cand = torch.tensor([-0.25, 0.0], dtype=torch.float32)
    out = _vector_alignment(ref, cand)
    assert out["cosine"] == -1.0
    assert out["projection_on_reference"] == -0.25
