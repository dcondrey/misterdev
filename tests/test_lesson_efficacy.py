"""Tier 3: lesson efficacy — reinforce on measured help, not mere recurrence."""

from misterdev.core.planning.lesson_store import (
    _MIN_EFFICACY_EVIDENCE,
    LessonStore,
)


def _store(tmp_path):
    return LessonStore(tmp_path / "lessons.json")


def test_recorded_lessons_get_stable_ids(tmp_path):
    store = _store(tmp_path)
    store.record(["always run black", "close the db in tests"])
    ids = sorted(le.id for le in store.retrieve_lessons())
    assert len(ids) == 2 and all(i > 0 for i in ids)
    assert len(set(ids)) == 2  # ids are distinct


def test_id_survives_text_refresh(tmp_path):
    store = _store(tmp_path)
    store.record(["always run black before committing"])
    original_id = store.retrieve_lessons()[0].id
    # A reworded restatement reinforces the SAME lesson; id must be preserved so
    # accumulated efficacy is not orphaned.
    store.record(["run black before you commit"])
    lessons = store.retrieve_lessons()
    assert len(lessons) == 1
    assert lessons[0].id == original_id


def test_credit_first_run_sets_baseline_without_delta(tmp_path):
    store = _store(tmp_path)
    store.record(["lesson A"])
    lid = store.retrieve_lessons()[0].id
    store.credit([lid], outcome=0.8)
    le = store.retrieve_lessons()[0]
    assert le.injected == 1
    assert le.efficacy == 0.0  # first run establishes baseline, no delta to credit


def test_helpful_lesson_accrues_positive_efficacy(tmp_path):
    store = _store(tmp_path)
    store.record(["helpful lesson"])
    lid = store.retrieve_lessons()[0].id
    store.credit([lid], outcome=0.5)  # baseline = 0.5
    store.credit([lid], outcome=0.9)  # above baseline -> positive delta
    le = store.retrieve_lessons()[0]
    assert le.efficacy > 0.0
    assert le.injected == 2


def test_harmful_lesson_is_quarantined_out_of_retrieval(tmp_path):
    store = _store(tmp_path)
    store.record(["good lesson", "bad lesson"])
    good = next(le for le in store.retrieve_lessons() if le.text == "good lesson")
    bad = next(le for le in store.retrieve_lessons() if le.text == "bad lesson")
    store.credit([good.id, bad.id], outcome=0.9)  # baseline 0.9
    # The bad lesson keeps riding along in far-below-baseline runs.
    for _ in range(_MIN_EFFICACY_EVIDENCE + 1):
        store.credit([bad.id], outcome=0.1)
    texts = [le.text for le in store.retrieve_lessons()]
    assert "good lesson" in texts
    assert "bad lesson" not in texts  # quarantined: enough evidence of harm
    # Still on disk, just not injected.
    bad_reloaded = next(
        le for le in LessonStore(store.path)._load()[0] if le.text == "bad lesson"
    )
    assert bad_reloaded.quarantined
    assert bad_reloaded.regress_hits >= _MIN_EFFICACY_EVIDENCE


def test_efficacy_boosts_ranking(tmp_path):
    store = _store(tmp_path)
    store.record(["lesson one", "lesson two"])
    one = next(le for le in store.retrieve_lessons() if le.text == "lesson one")
    two = next(le for le in store.retrieve_lessons() if le.text == "lesson two")
    store.credit([one.id, two.id], outcome=0.5)  # baseline
    # Only 'lesson two' proves helpful across several above-baseline runs.
    for _ in range(3):
        store.credit([two.id], outcome=0.95)
    ranked = [le.text for le in store.retrieve_lessons()]
    assert ranked[0] == "lesson two"


def test_quarantine_triggers_on_majority_regressions(tmp_path):
    # A lesson whose averaged efficacy hovers above the band but which is present
    # in a MAJORITY of below-baseline runs must still be quarantined (regress_hits
    # is read, not just stored).
    store = _store(tmp_path)
    store.record(["borderline lesson"])
    lid = store.retrieve_lessons()[0].id
    store.credit([lid], outcome=0.6)  # baseline 0.6
    # Alternate: big drops (regressions) then partial recoveries, so efficacy
    # oscillates around the band but regressions are the majority.
    store.credit([lid], outcome=0.4)  # regression
    store.credit([lid], outcome=0.75)  # above
    store.credit([lid], outcome=0.4)  # regression
    store.credit([lid], outcome=0.4)  # regression
    le = next(le for le in store._load()[0] if le.id == lid)
    assert le.regress_hits >= 3
    assert le.quarantined
    assert le.text not in [x.text for x in store.retrieve_lessons()]


def test_eviction_drops_quarantined_before_helpful(tmp_path):
    # A quarantined high-score lesson must be evicted before a proven-helpful
    # low-score one when the store is over capacity.
    from misterdev.core.planning.lesson_store import _retention_value

    store = _store(tmp_path)
    store.record(["harmful high-recurrence lesson"])
    store.record(["helpful lesson"])
    harmful = next(le for le in store._load()[0] if "harmful" in le.text)
    helpful = next(le for le in store._load()[0] if "helpful" in le.text)
    # Make harmful quarantined (many below-baseline runs) and helpful proven good.
    store.credit([harmful.id, helpful.id], outcome=0.8)  # baseline 0.8
    for _ in range(4):
        store.credit([harmful.id], outcome=0.1)
    for _ in range(4):
        store.credit([helpful.id], outcome=0.95)
    lessons = store._load()[0]
    h = next(le for le in lessons if "harmful" in le.text)
    g = next(le for le in lessons if "helpful" in le.text)
    # Retention ranks the helpful lesson above the quarantined one regardless of
    # raw score, so a cap would evict the harmful one first.
    assert _retention_value(g) > _retention_value(h)


def test_credit_is_noop_without_ids_or_outcome(tmp_path):
    store = _store(tmp_path)
    store.record(["lesson"])
    lid = store.retrieve_lessons()[0].id
    store.credit([], outcome=0.9)
    store.credit([lid], outcome=None)
    assert store.retrieve_lessons()[0].injected == 0


def test_legacy_file_backfills_ids(tmp_path):
    path = tmp_path / "lessons.json"
    path.write_text('["legacy one", "legacy two"]')
    store = LessonStore(path)
    lessons = store.retrieve_lessons()
    assert all(le.id > 0 for le in lessons)
    assert len({le.id for le in lessons}) == 2
