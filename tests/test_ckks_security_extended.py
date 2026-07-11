from system.ckks_security_extended import ATTACKS, MODELS, program
from system.ckks_security_transcript import EXPECTED_ESTIMATOR_COMMIT, extract_parameters


def test_extended_programs_pin_default_models_and_exclude_arora():
    parameters = extract_parameters()
    assert "arora_gb" not in ATTACKS
    for model in MODELS:
        for attack in ATTACKS:
            source = program(parameters, model, attack)
            assert EXPECTED_ESTIMATOR_COMMIT in source
            assert "COST_MODEL MATZOV" in source
            assert "SHAPE_MODEL GSA" in source
            assert f"ATTACK_BEGIN {attack}" in source
            assert "RC.ADPS16" not in source


def test_rough_and_extended_cost_models_are_not_conflated():
    source = program(extract_parameters(), "seal_exact_cbd21", "usvp")
    assert "RC.MATZOV" in source
    assert "extended/default-cost" in source
