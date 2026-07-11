from system.circuit_privacy_audit import implementation_trace, run


def test_trace_records_absent_sanitisation():
    trace = implementation_trace()
    assert trace["operation_presence"]["multiply_plain"]
    assert trace["operation_presence"]["rotation"]
    assert trace["explicit_output_sanitisation_present"] is False


def test_replay_diagnostic_is_deterministic_for_fixed_request():
    result = run(dimension=8, candidates=2, repeats=2, seed=4)
    fixed = result["identical_encrypted_request_replay"]
    fresh = result["fresh_client_encryption"]
    assert fixed["unique_response_hashes"] == 1
    assert fixed["all_serialized_responses_identical"] is True
    assert fresh["unique_response_hashes"] == 2
    assert max(fixed["max_abs_decryption_error"]) < 1e-3
    assert "not a circuit-privacy proof" in result["disclaimer"]
