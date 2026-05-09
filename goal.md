# Goal

## Project
cyberai — a Python framework for building AI-powered cybersecurity agents.

## Description
A lightweight, modular framework for creating AI agents that can perform security tasks through tool use, agent handoffs, and guardrails. The framework provides:
- Agent abstraction with instructions, tools, and configurable behavior
- Tool system with function decorators and JSON schema generation
- Handoff mechanism for agent-to-agent delegation
- Input/output guardrails for safety validation
- Runner that orchestrates agent execution loops
- Model interface abstraction for different LLM providers
- Token usage tracking and result handling
- Streaming support for real-time agent output

## Scope
- ~15 production source files to implement
- ~10 test files to write
- Reproduce core agent SDK functionality with tools, handoffs, guardrails, and runner
