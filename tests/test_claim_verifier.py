"""Tests for the independent completeness-claim verifier.

The verifier must be CONSERVATIVE: it drops a claim only on a positive refutation
with evidence; confirmation, "unsure", an unparseable verdict, no LLM, or a
timeout all KEEP the claim so genuine work is never silently lost.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from my_project_orchestrator.agent import ProjectOrchestrator
from my_project_orchestrator.core.assessment import ProjectAssessment
from my_project_orchestrator.core.claim_verifier import (
    CONFIRMED,
    KEPT,
    REFUTED,
    Claim,
    ClaimVerdict,
    verify_claims,
)
from my_project_orchestrator.core.modes import BuildMode
from my_project_orchestrator.core.report import BuildReport


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
        "my_project_orchestrator.core.claim_verifier.verify_claims", fake_verify
    )
    assessment = ProjectAssessment()
    assessment.features.stubs = ["intentional.rs", "real.rs"]
    report = _report(assessment)

    ProjectOrchestrator()._verify_completeness_claims(_project({}), assessment, report)

    assert assessment.features.stubs == ["real.rs"]
    assert any("intentional.rs" in d for d in report.key_decisions)


def test_pipeline_noop_when_disabled(monkeypatch):
    def boom(*a, **k):  # must not be called when the gate is off
        raise AssertionError("verifier ran while disabled")

    monkeypatch.setattr(
        "my_project_orchestrator.core.claim_verifier.verify_claims", boom
    )
    assessment = ProjectAssessment()
    assessment.features.stubs = ["intentional.rs"]
    report = _report(assessment)

    ProjectOrchestrator()._verify_completeness_claims(
        _project({"orchestrator": {"verify_claims": False}}), assessment, report
    )

    assert assessment.features.stubs == ["intentional.rs"]
