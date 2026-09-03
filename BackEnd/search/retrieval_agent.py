"""LLM-based retrieval strategy agent.

This module provides an agent layer that uses a large language model to
decide *how* to search for a given query, replacing the hardcoded intent
classification with a more flexible, context-aware decision.

Design principles:
- Non-destructive: the agent ENHANCES the existing hardcoded logic, never
  replaces it. When the LLM is unavailable, times out, or returns invalid
  output, the system falls back to the original hardcoded behavior.
- Configurable: agent behavior is controlled via environment variables.
- Auditable: the agent's reasoning and decisions are stored in the profile
  for debugging and analysis.
- Safe: all LLM output is sanitized and validated before being applied.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

AGENT_ENABLED = os.environ.get("RETRIEVAL_AGENT_ENABLED", "true").lower() in ("1", "true", "yes")
AGENT_TIMEOUT = float(os.environ.get("RETRIEVAL_AGENT_TIMEOUT", "3.0"))
AGENT_MODEL = os.environ.get("RETRIEVAL_AGENT_MODEL", os.environ.get("LLM_MODEL", "qwen-turbo"))
AGENT_TEMPERATURE = float(os.environ.get("RETRIEVAL_AGENT_TEMPERATURE", "0.1"))
AGENT_MAX_TOKENS = int(os.environ.get("RETRIEVAL_AGENT_MAX_TOKENS", "400"))
AGENT_MIN_CONFIDENCE = float(os.environ.get("RETRIEVAL_AGENT_MIN_CONFIDENCE", "0.5"))
AGENT_CACHE_TTL_SECONDS = max(0.0, float(os.environ.get("RETRIEVAL_AGENT_CACHE_TTL_SECONDS", "300")))
AGENT_CACHE_MAX_ITEMS = max(1, int(os.environ.get("RETRIEVAL_AGENT_CACHE_MAX_ITEMS", "256")))

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_API_KEY = DEEPSEEK_API_KEY or os.environ.get("QWEN_API_KEY", "")
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL",
    "https://api.deepseek.com" if DEEPSEEK_API_KEY else "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_THINKING_ENABLED = os.environ.get("LLM_THINKING_ENABLED", "false").lower() in ("1", "true", "yes", "enabled")
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "").strip().lower()


def _normalized_reasoning_effort() -> str:
    # DeepSeek Chat Completions maps the user-facing medium level to high.
    if LLM_REASONING_EFFORT == "medium":
        return "high"
    return LLM_REASONING_EFFORT

# All available retrieval channels
ALL_CHANNELS: Tuple[str, ...] = (
    "vector",
    "keyword",
    "heading",
    "structure",
    "article",
    "table",
    "target_document",
)

# Valid intent categories
VALID_INTENTS: Tuple[str, ...] = (
    "article_lookup",
    "comparison",
    "count",
    "table",
    "definition_or_explanation",
    "requirement",
    "procedure_or_method",
    "fault_diagnosis",
    "general",
)

# Valid query types
VALID_QUERY_TYPES: Tuple[str, ...] = (
    "article_lookup",
    "fault_diagnosis",
    "procedure",
    "concept_explanation",
    "general",
)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class AgentRetrievalPlan:
    """The retrieval plan produced by the LLM agent.

    Fields are optional — only fields that the agent explicitly decides
    to override are set. None means "use the hardcoded default".
    """

    # Core intent decision
    intent: Optional[str] = None
    query_type: Optional[str] = None

    # Channel selection: if None, use all channels (hardcoded behavior)
    enabled_channels: Optional[List[str]] = None

    # Recall budget adjustments (multipliers, e.g. 0.5 = half the default)
    recall_multipliers: Dict[str, float] = field(default_factory=dict)

    # Query expansion suggestions
    query_expansions: List[str] = field(default_factory=list)

    # Strategy flags
    use_hyde: Optional[bool] = None
    structure_first: Optional[bool] = None
    use_rerank: Optional[bool] = None

    # Agent metadata
    confidence: float = 0.0
    reasoning: str = ""
    used_agent: bool = False
    latency_ms: float = 0.0

    def is_channel_enabled(self, channel: str) -> bool:
        """Check if a channel is enabled. None means all enabled."""
        if self.enabled_channels is None:
            return True
        return channel in self.enabled_channels

    def disabled_channels(self) -> List[str]:
        """Return list of channels explicitly disabled by the agent."""
        if self.enabled_channels is None:
            return []
        return [ch for ch in ALL_CHANNELS if ch not in self.enabled_channels]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for storage in profile."""
        return {
            "intent": self.intent,
            "query_type": self.query_type,
            "enabled_channels": self.enabled_channels,
            "recall_multipliers": self.recall_multipliers,
            "query_expansions": self.query_expansions,
            "use_hyde": self.use_hyde,
            "structure_first": self.structure_first,
            "use_rerank": self.use_rerank,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "used_agent": self.used_agent,
            "latency_ms": self.latency_ms,
        }


# ============================================================================
# Agent prompt
# ============================================================================

AGENT_SYSTEM_PROMPT = """你是一个航空文档检索系统的智能检索策略Agent。你的任务是分析用户查询，决定最优的检索策略。

## 可用的检索通道
- vector: 向量语义检索（适合模糊查询、概念查询）
- keyword: 关键词匹配（适合精确术语、代码）
- heading: 标题检索（适合章节、条款定位）
- structure: 结构记录检索（适合条款计数、分类、定义）
- article: 条款精确检索（适合"第X条"、CCAR编号）
- table: 表格检索（适合清单、检查表、数值表）
- target_document: 目标文档限定（适合明确指定文档的查询）

## 意图分类
- article_lookup: 条款查找（"第X条"、"CCAR-121第X条"）
- comparison: 比较对比（"区别"、"差异"、"比较"）
- count: 计数统计（"多少章"、"共有多少条"）
- table: 表格查询（"表"、"清单"、"检查表"）
- definition_or_explanation: 定义解释（"是什么"、"定义"、"含义"）
- requirement: 要求条件（"申请条件"、"应当"、"必须"、"资格"）
- procedure_or_method: 程序方法（"如何"、"怎么"、"步骤"、"流程"）
- fault_diagnosis: 故障诊断（"故障"、"异常"、"失效"、"排查"）
- general: 通用查询

## 输出格式
请严格输出JSON，不要输出任何其他文字：
{
  "intent": "意图分类",
  "query_type": "查询类型(article_lookup/fault_diagnosis/procedure/concept_explanation/general)",
  "enabled_channels": ["需要启用的通道列表，null表示全部启用"],
  "recall_multipliers": {"通道名": 召回量倍数(0.3-2.0)},
  "query_expansions": ["查询扩展建议，用于别名检索"],
  "use_hyde": true/false/null,
  "structure_first": true/false/null,
  "use_rerank": true/false/null,
  "confidence": 0.0-1.0,
  "reasoning": "简要说明你的决策理由"
}

## 决策原则
1. 条款查找类查询：优先article、heading、structure通道，减少vector召回
2. 定义解释类查询：优先vector、heading通道，use_hyde=true
3. 程序方法类查询：优先vector、keyword、heading通道
4. 故障诊断类查询：优先keyword、vector通道，增加召回量
5. 比较对比类查询：需要更多候选，增加candidate召回，全部通道启用
6. 计数统计类查询：优先structure通道，可减少其他通道
7. 简短模糊查询：use_hyde=true，全部通道启用
8. 明确指定文档的查询：target_document通道权重提高
9. 置信度低于0.5时，enabled_channels设为null（全部启用，交给硬编码逻辑）
"""


# ============================================================================
# Agent implementation
# ============================================================================

class RetrievalAgent:
    """LLM-based retrieval strategy agent.

    The agent analyzes a query and produces an AgentRetrievalPlan that
    can override or augment the hardcoded retrieval strategy.

    Usage:
        agent = RetrievalAgent()
        plan = agent.plan(query)
        if plan.used_agent:
            # apply plan to retrieval policy
            ...
    """

    def __init__(
        self,
        llm_call: Optional[Callable[[str, str], str]] = None,
        timeout: float = AGENT_TIMEOUT,
        model: str = AGENT_MODEL,
        temperature: float = AGENT_TEMPERATURE,
        max_tokens: int = AGENT_MAX_TOKENS,
    ):
        self._llm_call = llm_call
        self.timeout = timeout
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._plan_cache: Dict[Tuple[str, str], Tuple[float, AgentRetrievalPlan]] = {}
        self._plan_cache_lock = threading.RLock()

    def _get_cached_plan(self, query: str, hardcoded_intent: str) -> Optional[AgentRetrievalPlan]:
        if AGENT_CACHE_TTL_SECONDS <= 0:
            return None
        key = (query.strip(), hardcoded_intent.strip())
        with self._plan_cache_lock:
            cached = self._plan_cache.get(key)
            if cached is None:
                return None
            created_at, plan = cached
            if time.monotonic() - created_at > AGENT_CACHE_TTL_SECONDS:
                self._plan_cache.pop(key, None)
                return None
            return plan

    def _cache_plan(self, query: str, hardcoded_intent: str, plan: AgentRetrievalPlan) -> None:
        if AGENT_CACHE_TTL_SECONDS <= 0:
            return
        key = (query.strip(), hardcoded_intent.strip())
        with self._plan_cache_lock:
            if len(self._plan_cache) >= AGENT_CACHE_MAX_ITEMS and key not in self._plan_cache:
                oldest_key = next(iter(self._plan_cache))
                self._plan_cache.pop(oldest_key, None)
            self._plan_cache[key] = (time.monotonic(), plan)

    def _get_client(self):
        """Lazy-initialize the OpenAI-compatible client."""
        if self._client is not None:
            return self._client
        if not LLM_API_KEY:
            return None
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
            return self._client
        except Exception as exc:
            logger.warning("Failed to initialize LLM client for retrieval agent: %s", exc)
            return None

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM, using custom callable if provided."""
        if self._llm_call is not None:
            return self._llm_call(system_prompt, user_prompt)

        client = self._get_client()
        if client is None:
            raise RuntimeError("LLM client not available")

        request_options: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if LLM_THINKING_ENABLED:
            request_options["extra_body"] = {"thinking": {"type": "enabled"}}
            effort = _normalized_reasoning_effort()
            if effort:
                request_options["reasoning_effort"] = effort
        else:
            request_options["temperature"] = self.temperature

        response = client.chat.completions.create(**request_options)
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_llm_output(raw: str) -> Optional[Dict[str, Any]]:
        """Parse and sanitize LLM JSON output."""
        if not raw:
            return None

        # Try to extract JSON from the output (in case of markdown wrapping)
        text = raw.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # Find the first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end <= start:
            return None

        try:
            data = json.loads(text[start:end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _sanitize_plan(data: Dict[str, Any]) -> AgentRetrievalPlan:
        """Sanitize and validate the parsed plan data."""
        plan = AgentRetrievalPlan()

        # Intent
        intent = str(data.get("intent", "") or "").strip()
        if intent in VALID_INTENTS:
            plan.intent = intent

        # Query type
        query_type = str(data.get("query_type", "") or "").strip()
        if query_type in VALID_QUERY_TYPES:
            plan.query_type = query_type

        # Enabled channels
        channels = data.get("enabled_channels")
        if channels is None:
            plan.enabled_channels = None  # Use all
        elif isinstance(channels, list):
            valid = [ch for ch in channels if isinstance(ch, str) and ch in ALL_CHANNELS]
            if valid:
                plan.enabled_channels = valid
            # If empty or all invalid, leave as None (use all)

        # Recall multipliers
        multipliers = data.get("recall_multipliers") or {}
        if isinstance(multipliers, dict):
            for ch, mult in multipliers.items():
                if ch in ALL_CHANNELS and isinstance(mult, (int, float)):
                    clamped = max(0.3, min(2.0, float(mult)))
                    plan.recall_multipliers[ch] = clamped

        # Query expansions
        expansions = data.get("query_expansions") or []
        if isinstance(expansions, list):
            plan.query_expansions = [
                str(e).strip() for e in expansions
                if isinstance(e, str) and 2 <= len(e.strip()) <= 100
            ][:6]

        # Boolean flags (None means "use default")
        for flag_name in ("use_hyde", "structure_first", "use_rerank"):
            value = data.get(flag_name)
            if isinstance(value, bool):
                setattr(plan, flag_name, value)

        # Confidence
        confidence = data.get("confidence", 0.0)
        if isinstance(confidence, (int, float)):
            plan.confidence = max(0.0, min(1.0, float(confidence)))

        # Reasoning
        reasoning = str(data.get("reasoning", "") or "").strip()
        plan.reasoning = reasoning[:500]

        return plan

    def plan(self, query: str, *, hardcoded_intent: str = "") -> AgentRetrievalPlan:
        """Generate a retrieval plan for the given query.

        Args:
            query: The user's search query.
            hardcoded_intent: The intent from the hardcoded classifier,
                used as context for the LLM.

        Returns:
            An AgentRetrievalPlan. If the agent is disabled, unavailable,
            or returns low-confidence output, plan.used_agent will be False
            and all override fields will be None (i.e., use hardcoded defaults).
        """
        empty_plan = AgentRetrievalPlan(used_agent=False)

        if not AGENT_ENABLED:
            return empty_plan
        if not query or not str(query).strip():
            return empty_plan

        cached_plan = self._get_cached_plan(query, hardcoded_intent)
        if cached_plan is not None:
            logger.debug("Using cached retrieval agent plan for query: %s", query[:80])
            return cached_plan

        start_time = time.time()

        try:
            user_prompt = (
                f"用户查询：{query}\n\n"
                f"硬编码意图分类结果：{hardcoded_intent or 'unknown'}\n\n"
                f"请分析该查询，输出最优检索策略。"
            )

            raw_output = self._call_llm(AGENT_SYSTEM_PROMPT, user_prompt)
            parsed = self._parse_llm_output(raw_output)

            if parsed is None:
                logger.debug("Retrieval agent returned unparseable output for query: %s", query[:80])
                return empty_plan

            plan = self._sanitize_plan(parsed)
            plan.latency_ms = (time.time() - start_time) * 1000

            # Low confidence: don't apply overrides
            if plan.confidence < AGENT_MIN_CONFIDENCE:
                logger.debug(
                    "Retrieval agent confidence %.2f below threshold %.2f for query: %s",
                    plan.confidence, AGENT_MIN_CONFIDENCE, query[:80],
                )
                plan.used_agent = False
                # Keep reasoning for debugging but don't apply
                self._cache_plan(query, hardcoded_intent, plan)
                return plan

            plan.used_agent = True
            self._cache_plan(query, hardcoded_intent, plan)
            logger.info(
                "Retrieval agent plan: intent=%s type=%s channels=%s disabled=%s confidence=%.2f latency=%.0fms",
                plan.intent, plan.query_type,
                plan.enabled_channels if plan.enabled_channels else "all",
                plan.disabled_channels(),
                plan.confidence, plan.latency_ms,
            )
            return plan

        except Exception as exc:
            latency = (time.time() - start_time) * 1000
            logger.debug("Retrieval agent failed (%.0fms): %s", latency, exc)
            return empty_plan


# ============================================================================
# Plan application
# ============================================================================

def apply_agent_plan_to_profile(profile: Dict[str, Any], plan: AgentRetrievalPlan) -> Dict[str, Any]:
    """Apply an agent plan to a retrieval profile dict.

    This modifies the profile in place and returns it. Only fields that
    the agent explicitly decides to override are changed.
    """
    if not plan.used_agent:
        profile["agent_plan"] = plan.to_dict()
        return profile

    # Override intent and query_type
    if plan.intent:
        profile["agent_intent"] = plan.intent
        # Don't override the contract intent directly — let the contract
        # layer decide. But store it for the policy builder.
    if plan.query_type:
        profile["query_type"] = plan.query_type

    # Structure first
    if plan.structure_first is not None:
        profile["structure_first"] = plan.structure_first

    # Channel selection
    if plan.enabled_channels is not None:
        profile["agent_enabled_channels"] = list(plan.enabled_channels)
        profile["agent_disabled_channels"] = plan.disabled_channels()

    # Recall multipliers
    if plan.recall_multipliers:
        profile["agent_recall_multipliers"] = dict(plan.recall_multipliers)

    # Query expansions
    if plan.query_expansions:
        existing = profile.get("agent_query_expansions", [])
        profile["agent_query_expansions"] = list(existing) + plan.query_expansions

    # HyDE
    if plan.use_hyde is not None:
        profile["agent_use_hyde"] = plan.use_hyde

    # Rerank
    if plan.use_rerank is not None:
        profile["agent_use_rerank"] = plan.use_rerank

    # Store full plan for debugging
    profile["agent_plan"] = plan.to_dict()

    return profile


def apply_agent_plan_to_recall_budget(
    budget: "RecallBudget",  # noqa: F821 - forward reference
    plan: AgentRetrievalPlan,
) -> "RecallBudget":  # noqa: F821
    """Apply agent recall multipliers to a RecallBudget.

    Returns a new RecallBudget with adjusted recall counts.
    """
    if not plan.used_agent or not plan.recall_multipliers:
        return budget

    from search.common import RecallBudget

    adjustments: Dict[str, int] = {}
    for channel, multiplier in plan.recall_multipliers.items():
        current = getattr(budget, channel, None)
        if current is not None and isinstance(current, int):
            adjustments[channel] = max(5, int(current * multiplier))

    if not adjustments:
        return budget

    return RecallBudget(
        vector=adjustments.get("vector", budget.vector),
        keyword=adjustments.get("keyword", budget.keyword),
        heading=adjustments.get("heading", budget.heading),
        structure=adjustments.get("structure", budget.structure),
        article=adjustments.get("article", budget.article),
        table=adjustments.get("table", budget.table),
        target_document=adjustments.get("target_document", budget.target_document),
        alias_limit=budget.alias_limit,
        candidate_min=budget.candidate_min,
        candidate_max=budget.candidate_max,
        rerank_window=budget.rerank_window,
    )


# ============================================================================
# Convenience: agent-enhanced policy builder
# ============================================================================

_default_agent: Optional[RetrievalAgent] = None
_agent_lock = None  # Will be initialized on first use


def get_retrieval_agent() -> RetrievalAgent:
    """Get or create the default RetrievalAgent singleton."""
    global _default_agent, _agent_lock
    if _default_agent is None:
        import threading
        if _agent_lock is None:
            _agent_lock = threading.Lock()
        with _agent_lock:
            if _default_agent is None:
                _default_agent = RetrievalAgent()
    return _default_agent


def agent_enhanced_retrieval_policy(
    profile: Dict[str, Any],
    base_policy_builder: Callable[[Dict[str, Any]], Any],
    agent: Optional[RetrievalAgent] = None,
) -> Tuple[Any, AgentRetrievalPlan]:
    """Build a retrieval policy, enhanced by the agent if available.

    This is the main integration point. It:
    1. Builds the base (hardcoded) policy
    2. Asks the agent for a plan
    3. Applies the agent plan to the profile and policy
    4. Returns the enhanced policy and the agent plan

    Args:
        profile: The retrieval profile dict.
        base_policy_builder: Callable that builds the base RetrievalPolicy.
        agent: Optional RetrievalAgent instance. Uses the default singleton
            if not provided.

    Returns:
        Tuple of (enhanced_policy, agent_plan).
    """
    # Build base policy first (always, even if agent will override)
    base_policy = base_policy_builder(profile)

    # Try agent enhancement
    if agent is None:
        agent = get_retrieval_agent()

    query = str(profile.get("original_query", "") or profile.get("query", ""))
    hardcoded_intent = str(
        (profile.get("retrieval_contract") or {}).get("intent")
        or profile.get("query_type", "")
    )

    plan = agent.plan(query, hardcoded_intent=hardcoded_intent)

    if not plan.used_agent:
        profile["agent_plan"] = plan.to_dict()
        return base_policy, plan

    # Apply plan to profile
    apply_agent_plan_to_profile(profile, plan)

    # Apply recall multipliers to policy
    from search.common import RetrievalPolicy, FusionPolicy
    enhanced_recall = apply_agent_plan_to_recall_budget(base_policy.recall, plan)

    enhanced_policy = RetrievalPolicy(
        query_type=plan.query_type or base_policy.query_type,
        contract_intent=plan.intent or base_policy.contract_intent,
        recall=enhanced_recall,
        fusion=base_policy.fusion,
    )

    return enhanced_policy, plan
