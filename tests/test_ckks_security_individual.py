import pytest

from system.ckks_security_individual import ATTACKS, MODELS, atomic_write_json, program
from system.ckks_security_transcript import EXPECTED_ESTIMATOR_COMMIT, extract_parameters


def test_individual_inputs_pin_models_attacks_and_commit():
    parameters = extract_parameters()
    for model in MODELS:
        for attack in ATTACKS:
            if model != "seal_exact_cbd21" and attack == "arora_gb":
                continue
            source = program(parameters, model, attack)
            assert EXPECTED_ESTIMATOR_COMMIT in source
            assert f"ATTACK_BEGIN {attack}" in source
            assert "m=oo" in source
            assert parameters["coeff_modulus_product_q"] in source


def test_arora_gb_is_not_misapplied_to_gaussian_sensitivity():
    with pytest.raises(ValueError):
        program(extract_parameters(), "sigma_3_2_sensitivity", "arora_gb")


def test_checkpoint_write_is_atomic(tmp_path):
    target = tmp_path / "checkpoint.json"
    atomic_write_json(target, {"status": "timed_out_partial_output_archived"})
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert not (tmp_path / "checkpoint.json.tmp").exists()
