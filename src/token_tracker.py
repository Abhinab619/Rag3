"""
Token Usage Tracker for RAG Bot
================================
Tracks token counts, API calls, and estimated costs for every LLM invocation.
Uses LangChain's callback system to intercept calls transparently.

Usage:
    from src.token_tracker import tracker, get_tracking_callbacks, request_tracker

    # Pass callbacks to any LLM call:
    chain.invoke(inputs, config={"callbacks": get_tracking_callbacks()})

    # Or track a full request:
    with request_tracker("user query here") as rt:
        answer = bot.chat(query)
    print(rt.summary())

    # Get cumulative stats:
    print(tracker.get_stats())
"""

import time
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)



USD_TO_INR = 85.0

MODEL_PRICING = {
    
    "llama-3.3-70b-versatile": {"input": 0.59 * USD_TO_INR, "output": 0.79 * USD_TO_INR},
    "llama-3.1-70b-versatile": {"input": 0.59 * USD_TO_INR, "output": 0.79 * USD_TO_INR},
    "llama-3.1-8b-instant":    {"input": 0.05 * USD_TO_INR, "output": 0.08 * USD_TO_INR},
    "llama-3.2-3b-preview":    {"input": 0.06 * USD_TO_INR, "output": 0.06 * USD_TO_INR},
    "llama-3.2-1b-preview":    {"input": 0.04 * USD_TO_INR, "output": 0.04 * USD_TO_INR},
    
    "mixtral-8x7b-32768":      {"input": 0.24 * USD_TO_INR, "output": 0.24 * USD_TO_INR},
    
    "gemma2-9b-it":            {"input": 0.20 * USD_TO_INR, "output": 0.20 * USD_TO_INR},
    
    "_default":                {"input": 0.50 * USD_TO_INR, "output": 0.50 * USD_TO_INR},
}


def _get_pricing(model_name: str) -> Dict[str, float]:
    """Get pricing for a model, with fallback to default."""
    return MODEL_PRICING.get(model_name, MODEL_PRICING["_default"])


@dataclass
class LLMCallRecord:
    """A single LLM API call record."""
    timestamp: str
    model: str
    caller: str  # Which component made the call (e.g., "StaticBot.chat", "RAGGrader")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_inr: float = 0.0
    latency_ms: float = 0.0
    request_id: Optional[str] = None  # Groups calls within a single user request

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "caller": self.caller,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_inr": round(self.cost_inr, 6),
            "latency_ms": round(self.latency_ms, 2),
            "request_id": self.request_id,
        }


class TokenUsageCallback(BaseCallbackHandler):
    """
    LangChain callback handler that captures token usage from every LLM call.
    Works with ChatGroq and any LangChain-compatible LLM that reports usage metadata.
    """

    def __init__(self, tracker: "TokenTracker", caller: str = "unknown", request_id: str = None):
        super().__init__()
        self.tracker = tracker
        self.caller = caller
        self.request_id = request_id
        self._start_time: Optional[float] = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self._start_time = time.perf_counter()

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: list, **kwargs) -> None:
        self._start_time = time.perf_counter()

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        latency_ms = 0.0
        if self._start_time:
            latency_ms = (time.perf_counter() - self._start_time) * 1000

        # Extract token usage from the response
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage", {})

        # Some providers put it in different places
        if not token_usage and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    gen_info = getattr(gen, "generation_info", {}) or {}
                    if "token_usage" in gen_info:
                        token_usage = gen_info["token_usage"]
                        break
                    # Groq sometimes puts usage in response_metadata
                    msg = getattr(gen, "message", None)
                    if msg:
                        resp_meta = getattr(msg, "response_metadata", {}) or {}
                        if "token_usage" in resp_meta:
                            token_usage = resp_meta["token_usage"]
                            break
                        usage = resp_meta.get("usage", {})
                        if usage:
                            token_usage = usage
                            break

        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)

        # Extract model name
        model = llm_output.get("model_name", "unknown")
        if model == "unknown":
            model = llm_output.get("model", "unknown")
        # Try from response metadata
        if model == "unknown" and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg:
                        resp_meta = getattr(msg, "response_metadata", {}) or {}
                        model = resp_meta.get("model_name", resp_meta.get("model", "unknown"))
                        if model != "unknown":
                            break

        # Calculate cost in INR
        pricing = _get_pricing(model)
        cost = (prompt_tokens * pricing["input"] / 1_000_000) + \
               (completion_tokens * pricing["output"] / 1_000_000)

        record = LLMCallRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            caller=self.caller,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_inr=cost,
            latency_ms=latency_ms,
            request_id=self.request_id,
        )

        self.tracker.add_record(record)

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        logger.warning(f"[TokenTracker] LLM error in {self.caller}: {error}")


class TokenTracker:
    """
    Central token usage tracker. Thread-safe singleton that accumulates stats.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self._lock = threading.Lock()
        self._records: List[LLMCallRecord] = []
        self._model_stats: Dict[str, Dict[str, Union[int, float]]] = {}
        self._caller_stats: Dict[str, Dict[str, Union[int, float]]] = {}
        self._total_calls: int = 0
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._start_time: str = datetime.now(timezone.utc).isoformat()

        # Optional: persist logs to disk
        self._log_dir = log_dir
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = log_dir / f"token_usage_{datetime.now().strftime('%Y%m%d')}.jsonl"
        else:
            self._log_file = None

    def add_record(self, record: LLMCallRecord) -> None:
        """Add a new call record (thread-safe)."""
        with self._lock:
            self._records.append(record)
            self._total_calls += 1
            self._total_tokens += record.total_tokens
            self._total_cost += record.cost_inr

            # Per-model stats
            if record.model not in self._model_stats:
                self._model_stats[record.model] = {
                    "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "cost_inr": 0.0, "total_latency_ms": 0.0
                }
            ms = self._model_stats[record.model]
            ms["calls"] += 1
            ms["prompt_tokens"] += record.prompt_tokens
            ms["completion_tokens"] += record.completion_tokens
            ms["total_tokens"] += record.total_tokens
            ms["cost_inr"] += record.cost_inr
            ms["total_latency_ms"] += record.latency_ms

            # Per-caller stats
            if record.caller not in self._caller_stats:
                self._caller_stats[record.caller] = {
                    "calls": 0, "total_tokens": 0, "cost_inr": 0.0
                }
            cs = self._caller_stats[record.caller]
            cs["calls"] += 1
            cs["total_tokens"] += record.total_tokens
            cs["cost_inr"] += record.cost_inr

            # Log to console
            logger.info(
                f"[TokenTracker] {record.caller} | {record.model} | "
                f"Tokens: {record.prompt_tokens}→{record.completion_tokens} "
                f"(total: {record.total_tokens}) | Cost: ₹{record.cost_inr:.4f} | "
                f"Latency: {record.latency_ms:.0f}ms"
            )

            # Persist to disk
            if self._log_file:
                try:
                    with open(self._log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record.to_dict()) + "\n")
                except Exception as e:
                    logger.error(f"[TokenTracker] Failed to write log: {e}")

    def get_stats(self) -> dict:
        """Get cumulative usage statistics."""
        with self._lock:
            model_summary = {}
            for model, stats in self._model_stats.items():
                avg_latency = stats["total_latency_ms"] / stats["calls"] if stats["calls"] else 0
                model_summary[model] = {
                    "calls": stats["calls"],
                    "prompt_tokens": stats["prompt_tokens"],
                    "completion_tokens": stats["completion_tokens"],
                    "total_tokens": stats["total_tokens"],
                    "cost_inr": round(stats["cost_inr"], 4),
                    "avg_latency_ms": round(avg_latency, 2),
                }

            caller_summary = {}
            for caller, stats in self._caller_stats.items():
                caller_summary[caller] = {
                    "calls": stats["calls"],
                    "total_tokens": stats["total_tokens"],
                    "cost_inr": round(stats["cost_inr"], 4),
                }

            return {
                "session_start": self._start_time,
                "currency": "INR",
                "usd_to_inr_rate": USD_TO_INR,
                "total_calls": self._total_calls,
                "total_tokens": self._total_tokens,
                "total_cost_inr": round(self._total_cost, 4),
                "by_model": model_summary,
                "by_component": caller_summary,
                "pricing_table_inr_per_1m_tokens": {
                    k: {"input": round(v["input"], 2), "output": round(v["output"], 2)}
                    for k, v in MODEL_PRICING.items() if k != "_default"
                },
            }

    def get_request_stats(self, request_id: str) -> dict:
        """Get stats for a specific request."""
        with self._lock:
            request_records = [r for r in self._records if r.request_id == request_id]
            if not request_records:
                return {"request_id": request_id, "calls": 0}

            total_tokens = sum(r.total_tokens for r in request_records)
            total_cost = sum(r.cost_inr for r in request_records)
            total_latency = sum(r.latency_ms for r in request_records)

            return {
                "request_id": request_id,
                "calls": len(request_records),
                "total_tokens": total_tokens,
                "total_cost_inr": round(total_cost, 4),
                "total_latency_ms": round(total_latency, 2),
                "breakdown": [r.to_dict() for r in request_records],
            }

    def get_recent_calls(self, limit: int = 20) -> List[dict]:
        """Get the most recent N call records."""
        with self._lock:
            return [r.to_dict() for r in self._records[-limit:]]

    def reset(self) -> None:
        """Reset all stats (useful for testing)."""
        with self._lock:
            self._records.clear()
            self._model_stats.clear()
            self._caller_stats.clear()
            self._total_calls = 0
            self._total_tokens = 0
            self._total_cost = 0.0
            self._start_time = datetime.now(timezone.utc).isoformat()


# ─── Global Singleton ────────────────────────────────────────────────────────
_log_dir = Path(__file__).resolve().parent.parent / "logs" / "token_usage"
tracker = TokenTracker(log_dir=_log_dir)


def get_tracking_callbacks(caller: str = "unknown", request_id: str = None) -> list:
    """
    Get a list of callback handlers to pass to any LangChain chain/LLM call.

    Args:
        caller: Name of the component (e.g., "StaticBot.chat", "RAGGrader")
        request_id: Optional request ID to group calls under a single user request

    Returns:
        List of callback handlers to pass as config={"callbacks": ...}
    """
    return [TokenUsageCallback(tracker, caller=caller, request_id=request_id)]


# ─── Request-Level Tracking Context Manager ──────────────────────────────────
_current_request_id = threading.local()


def get_current_request_id() -> Optional[str]:
    """Get the current request ID (if inside a request_tracker context)."""
    return getattr(_current_request_id, "value", None)


@contextmanager
def request_tracker(query: str = ""):
    """
    Context manager that assigns a unique request ID so all LLM calls
    within the block are grouped together.

    Usage:
        with request_tracker("user's question") as req_id:
            answer = bot.chat(query)
        stats = tracker.get_request_stats(req_id)
    """
    import uuid
    req_id = str(uuid.uuid4())[:8]
    _current_request_id.value = req_id
    logger.info(f"[TokenTracker] === Request {req_id} started: '{query[:80]}' ===")

    try:
        yield req_id
    finally:
        stats = tracker.get_request_stats(req_id)
        logger.info(
            f"[TokenTracker] === Request {req_id} complete: "
            f"{stats['calls']} calls, {stats['total_tokens']} tokens, "
            f"₹{stats.get('total_cost_inr', 0):.4f} ==="
        )
        _current_request_id.value = None
