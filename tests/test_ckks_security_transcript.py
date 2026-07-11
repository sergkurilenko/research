import math

from system.ckks_security_transcript import (
    EXPECTED_ESTIMATOR_COMMIT,
    extract_parameters,
    parse_rop_exponents,
    sage_program,
)


def test_extracts_exact_seal_moduli_and_noise_model():
    parameters = extract_parameters()
    assert parameters["poly_modulus_degree"] == 8192
    assert parameters["coeff_modulus_primes"] == [
        1152921504606748673,
        1099511480321,
        1152921504606830593,
    ]
    assert parameters["total_coeff_modulus_bits"] == 160
    assert parameters["seal_tc128_max_coeff_modulus_bits"] == 218
    assert parameters["seal_tc128_headroom_bits"] == 58
    assert math.isclose(
        parameters["error_distribution"]["exact_standard_deviation"],
        math.sqrt(10.5),
    )


def test_sage_program_pins_commit_and_unbounded_samples():
    program = sage_program(extract_parameters(), "rough")
    assert EXPECTED_ESTIMATOR_COMMIT in program
    assert "ND.CenteredBinomial(21" in program
    assert "ND.Uniform(-1, 1" in program
    assert "m=oo" in program
    assert "LWE.estimate.rough(params)" in program


def test_parse_costs_keeps_minimum_per_model():
    raw = """MODEL_BEGIN cbd
ATTACK usvp :: rop: ≈2^151.2, red: ≈2^151.2
ATTACK dual :: rop: ≈2^147.5, mem: ≈2^80
MODEL_END cbd
"""
    parsed = parse_rop_exponents(raw)
    assert parsed["cbd"]["minimum_rop_log2"] == 147.5
    assert parsed["cbd"]["rop_log2_by_attack"]["usvp"] == 151.2
