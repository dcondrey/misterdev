"""Tests for the independent completeness-claim verifier.

The verifier must be CONSERVATIVE: it drops a claim only on a positive refutation
with evidence; confirmation, "unsure", an unparseable verdict, no LLM, or a
timeout all KEEP the claim so genuine work is never silently lost.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from misterdev.agent import ProjectOrchestrator
from misterdev.core.planning.assessment import ProjectAssessment
from misterdev.core.verification.claim_verifier import (
    CONFIRMED,
    KEPT,
    REFUTED,
    Claim,
    ClaimVerdict,
    verify_claims,
)
from misterdev.core.modes import BuildMode
from misterdev.core.reporting.report import BuildReport


def _call_returning(mapping):
    """A verify_call seam that picks a response by the claim label in the prompt."""

    def call(prompt: str) -> str:
        for label, response in mapping.items():
            if label in prompt:
                return response
        return ""

    return call


def test_refutes_only_with_explicit_false():
    claims = [
        Claim(kind="stub", label="tract_backend.rs", evidence="degrades by design")
    ]
    call = _call_returning(
        {"tract_backend.rs": '{"real": false, "reason": "documented graceful degrade"}'}
    )
    [v] = verify_claims(claims, verify_call=call)
    assert v.status == REFUTED and v.refuted
    assert "graceful" in v.reason


def test_confirmed_claim_is_kept():
    claims = [
        Claim(
            kind="incomplete", label="PaymentFlow", evidence="raise NotImplementedError"
        )
    ]
    call = _call_returning({"PaymentFlow": '{"real": true, "reason": "no impl"}'})
    [v] = verify_claims(claims, verify_call=call)
    assert v.status == CONFIRMED and not v.refuted


def test_unsure_keeps_claim():
    claims = [Claim(kind="incomplete", label="Ambiguous")]
    call = _call_returning({"Ambiguous": '{"real": null, "reason": "cannot tell"}'})
    [v] = verify_claims(claims, verify_call=call)
    assert v.status == KEPT and not v.refuted


def test_unparseable_verdict_keeps_claim():
    claims = [Claim(kind="stub", label="weird.py")]
    call = _call_returning({"weird.py": "I think it's probably fine honestly"})
    [v] = verify_claims(claims, verify_call=call)
    assert v.status == KEPT and not v.refuted


def test_missing_real_key_keeps_claim():
    claims = [Claim(kind="stub", label="x.py")]
    call = _call_returning({"x.py": '{"reason": "no verdict field"}'})
    [v] = verify_claims(claims, verify_call=call)
    assert v.status == KEPT


def test_no_verifier_available_keeps_all():
    claims = [Claim(kind="stub", label="a.py"), Claim(kind="incomplete", label="B")]
    verdicts = verify_claims(claims, verify_call=None, llm_client=None)
    assert [v.status for v in verdicts] == [KEPT, KEPT]
    assert all(not v.refuted for v in verdicts)


def test_empty_claims_returns_empty():
    assert verify_claims([], verify_call=_call_returning({})) == []


def test_mixed_batch_drops_only_refuted():
    claims = [
        Claim(kind="stub", label="intentional.rs"),
        Claim(kind="incomplete", label="RealGap"),
    ]
    call = _call_returning(
        {
            "intentional.rs": '{"real": false, "reason": "platform no-op"}',
            "RealGap": '{"real": true, "reason": "unimplemented"}',
        }
    )
    verdicts = verify_claims(claims, verify_call=call)
    refuted = [v.claim.label for v in verdicts if v.refuted]
    assert refuted == ["intentional.rs"]


def test_parallel_path_preserves_order():
    # No model -> claims fan out across a thread pool; results must stay in order.
    claims = [Claim(kind="stub", label=f"f{i}.rs") for i in range(3)]
    call = _call_returning(
        {
            "f0.rs": '{"real": false}',
            "f1.rs": '{"real": true}',
            "f2.rs": '{"real": null}',
        }
    )
    verdicts = verify_claims(claims, verify_call=call)
    assert [v.claim.label for v in verdicts] == ["f0.rs", "f1.rs", "f2.rs"]
    assert [v.status for v in verdicts] == [REFUTED, CONFIRMED, KEPT]


def test_refute_with_empty_reason_is_honest():
    # A content-free refute must not be dressed up with a fabricated rationale.
    claims = [Claim(kind="stub", label="x.rs")]
    call = _call_returning({"x.rs": '{"real": false, "reason": ""}'})
    [v] = verify_claims(claims, verify_call=call)
    assert v.refuted
    assert "without a stated reason" in v.reason


def test_resolve_claim_file_rejects_generic_substring_match():
    # A generic token must not substring-match an unrelated file (the false-refute
    # bug): "backend" should not resolve to backend_registry.py.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "backend_registry.rs").write_text("fn register() {}\n")
        file_map = "backend_registry.rs: fn register, fn lookup"
        got = ProjectOrchestrator._resolve_claim_file(
            root, "Streaming backend incomplete", file_map
        )
        assert got is None


def test_resolve_claim_file_matches_camelcase_symbol():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tract_backend.rs").write_text("struct TractBackend;\n")
        file_map = "tract_backend.rs: struct TractBackend, fn embed"
        got = ProjectOrchestrator._resolve_claim_file(
            root, "TractBackend wasm inference", file_map
        )
        assert got == root / "tract_backend.rs"


# --- pipeline wiring: the gate must prune the assessment before decomposition ---


def _project(config):
    td = tempfile.mkdtemp()
    return SimpleNamespace(
        config=config, llm_client=object(), path=Path(td), topography=None, name="t"
    )


def _report(assessment):
    return BuildReport(BuildMode.COMPLETE, "t", assessment, datetime.now())


def test_pipeline_prunes_refuted_stub(monkeypatch):
    def fake_verify(claims, **kw):
        return [
            ClaimVerdict(
                c, REFUTED if c.label == "intentional.rs" else KEPT, reason="judged"
            )
            for c in claims
        ]

    monkeypatch.setattr(
        "misterdev.core.verification.claim_verifier.verify_claims", fake_verify
    )
    project = _project({})
    # The gate only verifies claims whose file has readable source.
    (project.path / "intentional.rs").write_text("//! degrades by design\nfn x() {}\n")
    (project.path / "real.rs").write_text("fn y() {}\n")
    assessment = ProjectAssessment()
    assessment.features.stubs = ["intentional.rs", "real.rs"]
    report = _report(assessment)

    ProjectOrchestrator()._verify_completeness_claims(project, assessment, report)

    assert assessment.features.stubs == ["real.rs"]
    assert any("intentional.rs" in d for d in report.key_decisions)


def test_pipeline_keeps_claim_with_no_readable_source(monkeypatch):
    # A stub whose file does not exist must NOT be verified (no source to judge)
    # and therefore never dropped — even if the verifier would refute it.
    def fake_verify(claims, **kw):
        raise AssertionError("verifier called on a claim with no source")

    monkeypatch.setattr(
        "misterdev.core.verification.claim_verifier.verify_claims", fake_verify
    )
    assessment = ProjectAssessment()
    assessment.features.stubs = ["does_not_exist.rs"]
    report = _report(assessment)

    ProjectOrchestrator()._verify_completeness_claims(_project({}), assessment, report)

    assert assessment.features.stubs == ["does_not_exist.rs"]


def test_pipeline_duplicate_labels_drop_only_refuted(monkeypatch):
    # Two stubs sharing a label-collision risk: identity-based pruning must drop
    # only the refuted one, never a kept claim that happens to share a name.
    def fake_verify(claims, **kw):
        return [
            ClaimVerdict(c, REFUTED if "a.rs" in c.label else KEPT, reason="judged")
            for c in claims
        ]

    monkeypatch.setattr(
        "misterdev.core.verification.claim_verifier.verify_claims", fake_verify
    )
    project = _project({})
    (project.path / "a.rs").write_text("//! intentional\nfn a() {}\n")
    (project.path / "b.rs").write_text("fn b() {}\n")
    assessment = ProjectAssessment()
    assessment.features.stubs = ["a.rs", "b.rs"]
    report = _report(assessment)

    ProjectOrchestrator()._verify_completeness_claims(project, assessment, report)

    assert assessment.features.stubs == ["b.rs"]


def test_pipeline_noop_when_disabled(monkeypatch):
    def boom(*a, **k):  # must not be called when the gate is off
        raise AssertionError("verifier ran while disabled")

    monkeypatch.setattr(
        "misterdev.core.verification.claim_verifier.verify_claims", boom
    )
    assessment = ProjectAssessment()
    assessment.features.stubs = ["intentional.rs"]
    report = _report(assessment)

    ProjectOrchestrator()._verify_completeness_claims(
        _project({"orchestrator": {"verify_claims": False}}), assessment, report
    )

    assert assessment.features.stubs == ["intentional.rs"]
