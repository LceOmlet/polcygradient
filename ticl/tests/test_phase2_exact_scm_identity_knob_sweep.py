from ticl.analysis.phase2_exact_scm_identity_knob_sweep import build_default_cases


def test_build_default_cases_contains_unique_named_sweep_cases():
    cases = build_default_cases()
    names = [str(item["name"]) for item in cases]
    assert len(cases) >= 6
    assert len(set(names)) == len(names)
    assert names[0] == "baseline"
    assert any(name == "tanh_low_init_low_noise" for name in names)
