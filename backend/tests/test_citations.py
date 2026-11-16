from app.services.citations import extract_source_ids, parse_risk_findings, resolve_citations


def test_extract_source_ids_formats():
    text = "Clause [S2] and [S1, S3]; also [Source: S2] [S4][s5] but not [Schedule A] or [1]."
    assert extract_source_ids(text) == ["S2", "S1", "S3", "S4", "S5"]


def test_resolve_flags_hallucinated_ids():
    valid, invalid = resolve_citations("See [S1] and [S9].", {"S1", "S2"})
    assert valid == ["S1"] and invalid == ["S9"]


def test_parse_risk_findings():
    output = """### Risk 1: Unilateral termination
**Severity:** High
**Source:** [S2]
**Analysis:** The Supplier may terminate "at any time for convenience" on 7 days notice.

### Risk 2: Uncited claim
**Severity:** medium
**Analysis:** Something without a source [S7].

### Overall Observations
The allocation of risk favours the Supplier [S1]."""
    risks = parse_risk_findings(output, {"S1", "S2"})
    assert len(risks) == 2
    first, second = risks
    assert first.title == "Unilateral termination"
    assert first.severity == "High"
    assert first.source_ids == ["S2"] and not first.citation_missing
    assert first.analysis.startswith("The Supplier may terminate")
    assert second.severity == "Medium"
    assert second.citation_missing and second.invalid_source_ids == ["S7"]
    assert "Overall" not in second.analysis


def test_risk_sources_prefer_claim_level_citations():
    output = """### Risk 1: Broad confidentiality
**Severity:** High
**Source:** [S1][S2][S3][S4][S5][S6]
**Analysis:** Information remains the Bank's property [S5] and must be returned on termination [S2]."""
    risk = parse_risk_findings(output, {f"S{i}" for i in range(1, 7)})[0]
    assert risk.source_ids == ["S5", "S2"]
