from my_project_orchestrator.core.advisor import recommend_work, _names
from my_project_orchestrator.core.assessment import ProjectAssessment


class _FakeLLM:
    def __init__(self, response="[]", raises=False):
        self.response = response
        self.raises = raises

    def generate_code(self, prompt, system_prompt=""):
        if self.raises:
            raise RuntimeError("llm down")
        return self.response


def test_recommend_work_parses_and_normalizes_work_type():
    llm = _FakeLLM(
        '[{"title": "Finish auth", "rationale": "half done", "work_type": "feature"},'
        ' {"title": "Clean it", "work_type": "bogus"},'
        ' {"rationale": "no title -> dropped"}]'
    )
    recs = recommend_work(ProjectAssessment(), llm)
    assert [r.title for r in recs] == ["Finish auth", "Clean it"]
    assert recs[0].work_type == "feature"
    assert recs[1].work_type == "complete"  # invalid type normalized


def test_recommend_work_returns_empty_on_llm_failure():
    assert recommend_work(ProjectAssessment(), _FakeLLM(raises=True)) == []


def test_recommend_work_empty_array():
    assert recommend_work(ProjectAssessment(), _FakeLLM("[]")) == []


def test_names_formats_and_caps():
    class _Feat:
        def __init__(self, name):
            self.name = name

    assert _names([]) == "none"
    assert _names([_Feat("auth"), _Feat("db")]) == "auth, db"
    many = _names([_Feat(f"f{i}") for i in range(20)])
    assert many.count(",") == 11  # capped at 12 items
