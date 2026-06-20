import pytest

from my_project_orchestrator.llm.client import (
    BaseLLMClient, LLMResponse, LLMUsage, LLMCallError,
    BudgetExceededError, create_llm_client,
)


class FakeLLMClient(BaseLLMClient):
    """Concrete subclass for testing the abstract base."""

    def __init__(self, responses=None, config=None):
        super().__init__(config or {"llm": {}, "build": {"budget": 10.0}})
        self._responses = list(responses or [])
        self._call_count = 0

    def _call(self, prompt, system_prompt):
        self._call_count += 1
        if self._responses:
            item = self._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return LLMResponse(content="default response", usage=LLMUsage())


def test_generate_returns_response():
    client = FakeLLMClient([LLMResponse(content="hello")])
    resp = client.generate("prompt")
    assert resp.content == "hello"


def test_generate_code_returns_string():
    client = FakeLLMClient([LLMResponse(content="code here")])
    assert client.generate_code("prompt") == "code here"


def test_usage_tracking():
    usage = LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, estimated_cost=0.01)
    client = FakeLLMClient([
        LLMResponse(content="r1", usage=usage),
        LLMResponse(content="r2", usage=usage),
    ])
    client.generate("p1")
    client.generate("p2")
    assert client.cumulative_usage.call_count == 2
    assert client.cumulative_usage.total_tokens == 300
    assert client.cumulative_usage.estimated_cost == pytest.approx(0.02)


def test_budget_exceeded():
    client = FakeLLMClient(config={"llm": {}, "build": {"budget": 0.01}})
    client.cumulative_usage.estimated_cost = 0.02
    with pytest.raises(BudgetExceededError):
        client.generate("prompt")


def test_budget_remaining():
    client = FakeLLMClient(config={"llm": {}, "build": {"budget": 5.0}})
    client.cumulative_usage.estimated_cost = 1.5
    assert client.budget_remaining == pytest.approx(3.5)


def test_budget_remaining_floor():
    client = FakeLLMClient(config={"llm": {}, "build": {"budget": 1.0}})
    client.cumulative_usage.estimated_cost = 5.0
    assert client.budget_remaining == 0.0


def test_retry_on_retryable_error():
    client = FakeLLMClient([
        LLMCallError("rate limit", retryable=True),
        LLMResponse(content="recovered"),
    ])
    resp = client.generate("prompt")
    assert resp.content == "recovered"
    assert client._call_count == 2


def test_no_retry_on_non_retryable():
    client = FakeLLMClient([
        LLMCallError("invalid request", retryable=False),
    ])
    with pytest.raises(LLMCallError):
        client.generate("prompt")
    assert client._call_count == 1


def test_max_retries_exhausted():
    client = FakeLLMClient([
        LLMCallError("timeout", retryable=True),
        LLMCallError("timeout", retryable=True),
        LLMCallError("timeout", retryable=True),
    ])
    with pytest.raises(LLMCallError):
        client.generate("prompt")
    assert client._call_count == 3


def test_llm_usage_defaults():
    u = LLMUsage()
    assert u.prompt_tokens == 0
    assert u.estimated_cost == 0.0
    assert u.call_count == 0


def test_llm_response_defaults():
    r = LLMResponse(content="test")
    assert r.model == ""
    assert r.finish_reason == ""


def test_llm_call_error_retryable():
    e = LLMCallError("rate limit", retryable=True)
    assert e.retryable is True
    assert "rate limit" in str(e)


def test_llm_call_error_not_retryable():
    e = LLMCallError("bad request")
    assert e.retryable is False


def test_create_llm_client_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported"):
        create_llm_client({"llm": {"provider": "nonexistent"}})
