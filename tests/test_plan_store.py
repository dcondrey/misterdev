"""Persisted plan proposals with an approval gate."""

from dataclasses import dataclass

from misterdev.core.planning import plan_store


@dataclass
class _Rec:
    title: str
    rationale: str
    work_type: str


def test_save_assigns_ids_and_unapproved(tmp_path):
    items = plan_store.save_plan(
        tmp_path,
        [
            _Rec("Add auth", "no login exists", "feature"),
            _Rec("Fix flaky test", "CI is red", "debug"),
        ],
    )
    assert [it["id"] for it in items] == ["P-001", "P-002"]
    assert all(it["approved"] is False for it in items)
    assert items[0]["work_type"] == "feature"
    # Persisted and reloadable.
    assert plan_store.load_plan(tmp_path) == items


def test_save_accepts_dicts_and_skips_titleless(tmp_path):
    items = plan_store.save_plan(
        tmp_path,
        [{"title": "Real"}, {"rationale": "no title"}],
    )
    assert len(items) == 1
    assert items[0]["title"] == "Real"
    assert items[0]["work_type"] == "complete"


def test_load_missing_returns_none(tmp_path):
    assert plan_store.load_plan(tmp_path) is None


def test_approve_all(tmp_path):
    plan_store.save_plan(tmp_path, [_Rec("a", "", "feature"), _Rec("b", "", "debug")])
    updated = plan_store.set_approval(tmp_path, approve_all=True)
    assert all(it["approved"] for it in updated)
    assert len(plan_store.approved_items(tmp_path)) == 2


def test_approve_and_reject_subset(tmp_path):
    plan_store.save_plan(
        tmp_path,
        [_Rec("a", "", "feature"), _Rec("b", "", "debug"), _Rec("c", "", "feature")],
    )
    plan_store.set_approval(tmp_path, item_ids=["P-001", "P-003"])
    assert {it["id"] for it in plan_store.approved_items(tmp_path)} == {
        "P-001",
        "P-003",
    }
    # Rejecting one flips it back.
    plan_store.set_approval(tmp_path, reject_ids=["P-001"])
    assert {it["id"] for it in plan_store.approved_items(tmp_path)} == {"P-003"}


def test_reject_wins_a_tie(tmp_path):
    plan_store.save_plan(tmp_path, [_Rec("a", "", "feature")])
    updated = plan_store.set_approval(
        tmp_path, item_ids=["P-001"], reject_ids=["P-001"]
    )
    assert updated[0]["approved"] is False


def test_set_approval_without_plan_returns_none(tmp_path):
    assert plan_store.set_approval(tmp_path, approve_all=True) is None


def test_approved_items_empty_without_plan(tmp_path):
    assert plan_store.approved_items(tmp_path) == []
