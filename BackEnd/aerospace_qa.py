"""
aerospace_qa - Compatibility entry point.
Re-exports all public APIs from the refactored modules.

Original module was split into:
  - qa_prompt: PDF context enrichment, prompt building, LLM calling
  - qa_evidence: evidence quality scoring, multi-target coverage, selection
  - qa_response: highlight normalization, source formatting, abstain payloads
  - qa_structured: article counting, document applicability, direct answers
  - qa_entry: ask(), ask_structured() main entry points
"""

from qa.prompt import (
    build_context,
    build_history_context,
    build_user_prompt,
    call_qwen,
    prepare_prompt_evidence,
)

from qa.entry import (
    ask,
    ask_structured,
    main,
)

# Re-export modules themselves for attribute-style access
import qa.prompt
import qa.evidence
import qa.response
import qa.structured
import qa.entry

__all__ = [
    "ask",
    "ask_structured",
    "build_context",
    "build_history_context",
    "build_user_prompt",
    "call_qwen",
    "main",
    "prepare_prompt_evidence",
]
