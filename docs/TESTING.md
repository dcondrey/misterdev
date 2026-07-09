# Test suite

**1,880 tests pass, 4 skipped** across **125 test files** (`pytest -q`). Every
gate, seam, and orchestration path is covered by deterministic, offline-runnable
tests — the same discipline misterdev applies to the code it edits. The 4 skips
are key-gated live-integration tests (they run only when an API key is present).

```bash
uv run pytest -q            # full suite
uv run pytest tests/test_failure_view.py -q   # one area
uv run pytest --collect-only -q               # list every test
```

## Coverage by subsystem

Counts are tests per file (`pytest --collect-only`).

### Orchestration & the execution loop — 288
| File | Tests | Covers |
|---|--:|---|
| test_executor.py | 108 | the try-edit-verify loop, gate flow, guards (dangling-ref, tamper), no-edit escalation, certainty completion |
| test_orchestrator_fixes.py | 94 | regression tests for orchestration bugs |
| test_gatekeeper.py | 50 | the ordered gate runner (build/lint/test/typecheck + optional) |
| test_orchestrator_config.py | 28 | config resolution and gate enablement |
| test_gates_error_context.py | 7 | retry-context assembly + failure escalation |
| test_integration_gate_count.py | 16 | the wave integration gate + bisect-revert |
| test_budget_halt.py / test_progress.py / test_change_tracker.py / test_executor_git.py / test_reflection.py / test_scratchpad.py / test_preflight.py | 24 | budget enforcement, progress, git, reflection, preflight |

### Observation & failure analysis — 91
| File | Tests | Covers |
|---|--:|---|
| test_error_classifier.py | 36 | error classification |
| test_failure_view.py | 16 | exact-assertion parsers for **8 runners** (pytest/jest/cargo/xctest/dotnet/vitest/gotest/junit) |
| test_error_resolver.py | 15 | error→location attribution |
| test_failure_taxonomy.py | 10 | cause classification (artifact/observation/search/convergence/saturation) |
| test_probe.py | 22 | failure-triggered single-test isolation + `should_auto_probe` |
| test_wiring_probe_staging.py | 7 | probe + verifier-decomposition wiring |
| test_error_log_compressor.py | 7 | structure-aware error compression |
| test_failure_log.py | 14 | the durable failure stream |

### Gates & verification — 210
| File | Tests | Covers |
|---|--:|---|
| test_validator.py | 37 | code/syntax validation gate |
| test_web_verify.py | 35 | headless-browser web gate |
| test_critic.py | 28 | adversarial-critic gate |
| test_goal_check.py (+_wiring) | 35 | goal-check gate |
| test_vision_verify.py | 22 | vision verification gate |
| test_mutation_gate.py | 22 | mutation-score gate |
| test_claim_verifier.py | 16 | completeness-claim verification |
| test_runtime.py | 15 | runtime-smoke gate |
| test_audit_gate.py / test_audit.py / test_outcomes.py / test_security.py | 28 | audit gate, gate outcomes, security |

### Context, topography & contracts — 190
| File | Tests | Covers |
|---|--:|---|
| test_topography.py (+_cache) | 56 | the tree-sitter symbol graph |
| test_contracts_*.py (ts/cpp/swift/kotlin/js/c/csharp) + test_contracts.py | 88 | public-API contract extraction per language |
| test_lsp.py / test_lsp_swift.py / test_lsp_context.py / test_lsp_integration.py | 32 | LSP diagnostics incl. the sourcekit-lsp session |
| test_embeddings.py / test_context_budget.py | 22 | semantic ranking + context budgeting |

### Planning & decomposition — 90
| File | Tests | Covers |
|---|--:|---|
| test_assessment.py | 32 | project assessment model |
| test_decomposer.py | 15 | task decomposition + staging hint |
| test_verifier_decomposition.py | 18 | dense-reward staging (name + call-edge ordering) |
| test_targets.py / test_modes.py / test_grounded_spec.py / test_advisor.py | 25 | multi-target routing, modes, grounded spec |

### Models & LLM — 195
| File | Tests | Covers |
|---|--:|---|
| test_llm_client.py | 86 | client, retry, failover, budget |
| test_responses.py | 38 | SEARCH/REPLACE parsing + apply |
| test_model_selector.py / test_model_ledger.py / test_model_catalog.py / test_dynamic_model_integration.py / test_free_models.py | 76 | cost-aware selection, ledger, free-model routing |
| test_llm_cache.py / test_prompt_manager.py | 18 | response cache + prompts |

### Self-improvement (evolution + learning) — 168
| File | Tests | Covers |
|---|--:|---|
| test_evolution_*.py (guardrail/loop/fitness/proposer/archive/screen/holdout/attribution/adapters/driver/sandbox/prior + holdout_wiring) | 105 | the keep-if-better loop, held-out anti-overfit gate, reward-hacking guardrail, mutation proposer |
| test_lesson_store.py / test_lesson_efficacy.py / test_metacognition.py / test_learning_semantic.py | 45 | scored lesson memory + metacognition |
| test_reproduction_corpus.py / test_failure_targeting.py / test_sovereign.py | 18 | reproduction corpus, real-failure targeting |

### Guidance (per-language rules) — 46
`test_guidance.py` + `test_guidance_{rust,react,typescript,swift,python,kotlin,html,elixir,css,csharp,cpp}.py` — the relevance-selected best-practice rule engine.

### Harnesses — 30
| File | Tests | Covers |
|---|--:|---|
| test_polyglot_harness.py | 11 | the Exercism polyglot benchmark harness (rust/python/js/…) |
| test_native_harness.py | 14 | the swift/c# validation harness |
| test_swebench_harness.py | 5 | the SWE-bench harness |

### MCP, plugins & extensibility — 83
`test_mcp*.py` (59), `test_plugins.py`, `test_extensibility.py`, `test_dependency_tool.py`, `test_command_tool.py`, `test_nl_cli.py`.

### Reporting, config & infrastructure — 209
`test_governance.py` (127), `test_report*.py`, `test_config.py`, `test_registry.py`, `test_container.py`, `test_spec_tests.py`, `test_task_manager.py`, `test_agent_helpers.py`, `test_bounded.py`, `test_process.py`, `test_file_utils.py`, `test_gitcmd.py`, `test_detect_lint_deps.py`, `test_hms_integration.py`, `test_independent.py`.

## Notes

- **Deterministic & offline.** Parsers are validated against *captured real*
  runner output; LLM/subprocess/network boundaries are injected or mocked. No
  test spends money or needs a toolchain to run.
- **Skips are honest.** The 4 skipped tests are live-integration tests gated on
  an API key; they report `skipped`, never a silent pass.
- **Empirical validation** (benchmark solve rates) is separate — see
  [benchmark-results.md](benchmark-results.md). The unit suite proves the
  *harness*; the benchmark proves the *capability*.
