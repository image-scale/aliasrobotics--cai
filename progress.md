# Progress

## Round 1
**Task**: Task 1 — Core exceptions and usage tracking
**Files created**: src/cyberai/__init__.py, src/cyberai/exceptions.py, src/cyberai/usage.py, tests/test_exceptions.py, tests/test_usage.py, tests/conftest.py, pyproject.toml
**Commit**: Add a foundation for error handling and token usage tracking in the agent framework
**Acceptance**: 8/8 criteria met
**Verification**: tests FAIL on previous state, PASS on current state

## Round 2
**Task**: Task 2 — Model settings and run context
**Files created**: src/cyberai/model_settings.py, src/cyberai/run_context.py, tests/test_model_settings.py, tests/test_run_context.py
**Commit**: Add model configuration settings and a run context wrapper for the agent framework
**Acceptance**: 6/6 criteria met
**Verification**: tests FAIL on previous state, PASS on current state

## Round 3
**Task**: Task 3 — Function tool system
**Files created**: src/cyberai/tool.py, tests/test_tool.py
**Commit**: Add a function tool system that converts Python functions into LLM-callable tools
**Acceptance**: 10/10 criteria met
**Verification**: tests FAIL on previous state, PASS on current state

## Round 4
**Task**: Task 4 — Run items and model response types
**Files created**: src/cyberai/items.py, tests/test_items.py
**Commit**: Add run item types and model response handling for tracking agent execution
**Acceptance**: 8/8 criteria met
**Verification**: tests FAIL on previous state, PASS on current state
