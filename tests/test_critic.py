import time

from misterdev.core.verification.critic import (
    APPROVED,
    REJECTED,
    SKIP,
    CritiqueVerdict,
    run_edit_critic,
    _parse_verdict,
    _render_candidate,
    _default_critic_call,
    _aggregate_panel,
)


_EDIT = {"src/a.py": "def f():\n    return 1\n"}


# --- SKIP semantics (the critic must never block on its own limits) ---------


def test_skip_when_no_candidate_edit():
    res = run_edit_critic("task", "criteria", {}, critic_call=lambda p: '{"approved": true}')
    assert res.status == SKIP
    assert "no candidate" in res.reason


def test_skip_when_no_critic_or_client():
    res = run_edit_critic("task", "criteria", _EDIT, llm_client=None)
    assert res.status == SKIP
    assert "no LLM critic" in res.reason


def test_skip_on_unparseable_verdict():
    res = run_edit_critic("t", "c", _EDIT, critic_call=lambda p: "looks fine to me")
    assert res.status == SKIP
    assert "unparseable" in res.reason


def test_skip_on_non_boolean_approved():
    res = run_edit_critic("t", "c", _EDIT, critic_call=lambda p: '{"approved": "yes"}')
    assert res.status == SKIP


def test_skip_on_critic_error():
    def boom(_prompt):
        raise RuntimeError("model down")

    res = run_edit_critic("t", "c", _EDIT, critic_call=boom)
    assert res.status == SKIP
    assert "error" in res.reason


def test_skip_on_timeout():
    def slow(_prompt):
        time.sleep(5)
        return '{"approved": true}'

    res = run_edit_critic("t", "c", _EDIT, critic_call=slow, timeout=0.2)
    assert res.status == SKIP
    assert "timed out" in res.reason


# --- verdicts ---------------------------------------------------------------


def test_approved_verdict():
    res = run_edit_critic("t", "c", _EDIT, critic_call=lambda p: '{"approved": true}')
    assert res.status == APPROVED
    assert res.approved
    assert res.objections == []


def test_rejected_with_objections():
    raw = '{"approved": false, "objections": ["no null check", "leaks a file handle"]}'
    res = run_edit_critic("t", "c", _EDIT, critic_call=lambda p: raw)
    assert res.status == REJECTED
    assert res.rejected
    assert "no null check" in res.objections
    assert len(res.objections) == 2


def test_rejected_without_objections_gets_generic_one():
    res = run_edit_critic(
        "t", "c", _EDIT, critic_call=lambda p: '{"approved": false, "objections": []}'
    )
    assert res.status == REJECTED
    assert len(res.objections) == 1


def test_verdict_tolerates_prose_and_fence():
    raw = 'Here is my review:\n```json\n{"approved": false, "objections": ["bug"]}\n```'
    res = _parse_verdict(raw)
    assert res.status == REJECTED
    assert res.objections == ["bug"]


def test_parse_empty_is_skip():
    assert _parse_verdict("").status == SKIP


# --- candidate rendering ----------------------------------------------------


def test_render_candidate_includes_paths_sorted():
    out = _render_candidate({"b.py": "y = 2", "a.py": "x = 1"})
    assert out.index("a.py") < out.index("b.py")
    assert "x = 1" in out and "y = 2" in out


def test_render_candidate_truncates_large_files():
    big = {"big.py": "x" * 50000}
    out = _render_candidate(big)
    assert "truncated" in out
    assert len(out) < 50000


# --- independent-model selection (the principle's core) ---------------------


class _FakeClient:
    def __init__(self):
        self.calls = []
        self.entered_model = None
        self._active = None

    def generate_code(self, prompt, system=""):
        self.calls.append(self._active)
        return '{"approved": true}'

    class _Ctx:
        def __init__(self, client, model):
            self._client = client
            self._model = model

        def __enter__(self):
            self._client._active = self._model
            self._client.entered_model = self._model
            return self._client

        def __exit__(self, *a):
            self._client._active = None
            return False

    def with_model(self, model):
        return _FakeClient._Ctx(self, model)


def test_uses_independent_model_when_configured():
    client = _FakeClient()
    res = run_edit_critic("t", "c", _EDIT, llm_client=client, critic_model="other/model")
    assert res.status == APPROVED
    # The critic call ran under the independent model, not the generator's.
    assert client.entered_model == "other/model"
    assert client.calls == ["other/model"]


def test_falls_back_to_generator_model_without_critic_model():
    client = _FakeClient()
    res = run_edit_critic("t", "c", _EDIT, llm_client=client, critic_model=None)
    assert res.status == APPROVED
    # No model switch happened; the call ran on the generator's own model.
    assert client.entered_model is None
    assert client.calls == [None]


def test_default_critic_call_none_without_client():
    assert _default_critic_call(None, "m") is None


class _NoSwitchClient:
    def generate_code(self, prompt, system=""):
        return '{"approved": true}'


def test_critic_model_set_but_client_cannot_switch_still_runs():
    # No with_model on the client -> runs on its own model rather than failing.
    res = run_edit_critic(
        "t", "c", _EDIT, llm_client=_NoSwitchClient(), critic_model="other/model"
    )
    assert res.status == APPROVED


def test_verdict_repr_is_compact():
    v = CritiqueVerdict(REJECTED, objections=["a", "b"])
    assert "rejected" in repr(v) and "2" in repr(v)


# --- diff-aware rendering ---------------------------------------------------


def test_render_candidate_uses_diffs_when_given():
    diffs = {"a.py": "@@ -1 +1 @@\n-x = 1\n+x = 2\n"}
    out = _render_candidate({"a.py": "x = 2\n"}, diffs)
    assert "unified diffs" in out
    assert "+x = 2" in out


def test_render_candidate_falls_back_to_content():
    out = _render_candidate({"a.py": "x = 1\n"})
    assert "full content" in out
    assert "x = 1" in out


def test_run_edit_critic_with_diffs_passes_them_to_call():
    seen = {}

    def call(prompt):
        seen["prompt"] = prompt
        return '{"approved": true}'

    res = run_edit_critic(
        "t",
        "c",
        {"a.py": "x = 2\n"},
        critic_call=call,
        candidate_diffs={"a.py": "@@ -1 +1 @@\n-x = 1\n+x = 2\n"},
    )
    assert res.status == APPROVED
    assert "+x = 2" in seen["prompt"]


# --- panel aggregation ------------------------------------------------------


def test_aggregate_panel_majority_rejects():
    verdicts = [
        CritiqueVerdict(REJECTED, objections=["bug A"]),
        CritiqueVerdict(REJECTED, objections=["bug B"]),
        CritiqueVerdict(APPROVED),
    ]
    out = _aggregate_panel(verdicts)
    assert out.status == REJECTED
    assert set(out.objections) == {"bug A", "bug B"}


def test_aggregate_panel_tie_approves():
    verdicts = [CritiqueVerdict(REJECTED, objections=["x"]), CritiqueVerdict(APPROVED)]
    assert _aggregate_panel(verdicts).status == APPROVED


def test_aggregate_panel_all_skip_is_skip():
    assert _aggregate_panel([CritiqueVerdict(SKIP), CritiqueVerdict(SKIP)]).status == SKIP


def test_aggregate_panel_dedupes_objections():
    verdicts = [
        CritiqueVerdict(REJECTED, objections=["dup", "a"]),
        CritiqueVerdict(REJECTED, objections=["dup", "b"]),
    ]
    out = _aggregate_panel(verdicts)
    assert out.objections.count("dup") == 1


def test_panel_runs_multiple_members_and_rejects_on_majority():
    # Three members, all reject (the fake returns a rejection regardless of lens).
    calls = {"n": 0}

    def call(prompt):
        calls["n"] += 1
        return '{"approved": false, "objections": ["leak"]}'

    res = run_edit_critic("t", "c", {"a.py": "x"}, critic_call=call, panel=3)
    assert res.status == REJECTED
    assert calls["n"] == 3


def test_panel_member_error_is_abstention_not_failure():
    # Two reject, one raises -> abstention -> majority still rejects.
    state = {"n": 0}

    def call(prompt):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("flaky member")
        return '{"approved": false, "objections": ["bug"]}'

    res = run_edit_critic("t", "c", {"a.py": "x"}, critic_call=call, panel=3)
    assert res.status == REJECTED


def test_critic_checks_root_cause_and_dry():
    # The critic is the structural home for symptom-vs-root-cause and DRY: both
    # must be present in the prompt and the panel lenses.
    from misterdev.core.verification.critic import _PROMPT, _LENSES

    assert "ROOT CAUSE" in _PROMPT and "DRY" in _PROMPT
    lenses = " ".join(_LENSES)
    assert "ROOT CAUSE" in lenses and "DUPLICATION" in lenses
