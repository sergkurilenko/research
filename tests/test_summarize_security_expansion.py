import json
from pathlib import Path

from system.summarize_security_expansion import summarize


ROOT = Path(__file__).resolve().parents[1]
EXPANSION = ROOT / "results" / "system_revision" / "security_expansion"
SECURITY = EXPANSION / "lwe_security"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_summary_preserves_completed_and_incomplete_security_evidence():
    result = summarize(
        parameters_record=_load(SECURITY / "security_parameters.json"),
        rough_record=_load(SECURITY / "individual_security_estimates.json"),
        extended_record=_load(SECURITY / "extended_security_estimates.json"),
        circuit_record=_load(EXPANSION / "circuit_privacy_boundary.json"),
        million_record=_load(EXPANSION / "candidate_id_million.json"),
        beir_record=_load(EXPANSION / "candidate_id_beir_confirmatory.json"),
    )
    assert result["standard_rough_batch"]["status"] == (
        "timed_out_partial_output_archived"
    )
    assert result["standard_rough_batch"]["numerical_estimate_available"] is False
    for model in ("seal_exact_cbd21", "sigma_3_2_sensitivity"):
        assert result["rough_individual"]["models"][model][
            "minimum_completed_rop_log2"
        ] == 146.9
        assert result["extended_individual"]["models"][model][
            "minimum_completed_rop_log2"
        ] == 175.8
    assert result["rough_individual"]["models"]["seal_exact_cbd21"][
        "arora_gb"
    ]["status"] == "timed_out_partial_output_archived"
    assert result["extended_individual"]["models"]["seal_exact_cbd21"]["bkw"][
        "status"
    ] == "timed_out_partial_output_archived"


def test_summary_keeps_privacy_boundary_and_confirmatory_counts():
    result = summarize(
        parameters_record=_load(SECURITY / "security_parameters.json"),
        rough_record=_load(SECURITY / "individual_security_estimates.json"),
        extended_record=_load(SECURITY / "extended_security_estimates.json"),
        circuit_record=_load(EXPANSION / "circuit_privacy_boundary.json"),
        million_record=_load(EXPANSION / "candidate_id_million.json"),
        beir_record=_load(EXPANSION / "candidate_id_beir_confirmatory.json"),
    )
    assert result["candidate_id_leakage"]["million_query_count"] == 400
    assert result["candidate_id_leakage"]["beir_confirmatory_query_count"] == 3230
    assert result["circuit_privacy_boundary"][
        "explicit_output_sanitisation_present"
    ] is False
    assert result["circuit_privacy_boundary"][
        "formal_circuit_privacy_established"
    ] is False
