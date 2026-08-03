"""
wayfound_monitor.py — Wayfound agent-supervision hooks for the query app.

Sends each query-app run to Wayfound (app.wayfound.ai) as a session so the
supervisor can review agent behavior end to end: the user's query, the
planner's routing decision, every agent's output (with self-critique rounds),
and any revision loop that follows.

Design rules:
    - Fire-and-forget: uploads run on background threads and can never block
      or break the UI. Any upload failure disables recording for that run only.
    - Zero-config off switch: if WAYFOUND_API_KEY is unset or the `wayfound`
      package is not installed, every hook is a silent no-op.
    - One Wayfound session per user query. Section revisions and
      validation-agent runs APPEND to the same session, so the whole feedback
      loop reads as one transcript in the supervisor.

Engine only — all UI lives in query_app.py. multiagent.py is untouched: the
orchestrator's result dict already carries a completion timestamp per agent,
which is enough to reconstruct the run timeline without instrumenting agents.

Configuration (.env):
    WAYFOUND_API_KEY   — required to enable; ask a Wayfound admin to create one
    WAYFOUND_AGENT_ID  — optional; defaults to the registered "MongoDB" agent
"""

import os
import threading
from datetime import datetime, timezone

# The "MongoDB" agent registered in Wayfound (Agents → connection page).
DEFAULT_AGENT_ID = "fb0d89cf-f0c1-4936-b3a9-9dd9bf7b315a"

# Wayfound analyzes full transcripts, but bound single messages so a giant
# retrieval or table block can't balloon the payload.
_MAX_CONTENT_CHARS = 12000

# Result-dict keys that hold an agent section, in pipeline order.
_SECTIONS = ("market", "performance", "risk", "stress_test", "newsletter")

_SECTION_LABELS = {
    "market": "Market Context Analysis",
    "performance": "Portfolio Performance Analysis",
    "risk": "Risk Analysis",
    "stress_test": "Stress Test",
    "newsletter": "Investor Newsletter",
}

_availability = None  # cached: None = not checked yet


# ============================================================
# Availability
# ============================================================

def utc_now_iso() -> str:
    """Current UTC time in the ISO-8601 'Z' format Wayfound expects."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def wayfound_available() -> bool:
    """True when the SDK is installed and an API key is configured."""
    global _availability
    if _availability is None:
        if not os.getenv("WAYFOUND_API_KEY"):
            _availability = False
        else:
            try:
                import wayfound  # noqa: F401
                _availability = True
            except ImportError:
                print(
                    "Wayfound: WAYFOUND_API_KEY is set but the SDK is not "
                    "installed — `pip install wayfound` to enable supervision."
                )
                _availability = False
    return _availability


# ============================================================
# Recorder — one Wayfound session per user query
# ============================================================

class WayfoundRecorder:
    """
    Owns one Wayfound session. record() returns immediately; uploads run on
    background daemon threads, serialized by a lock so the session is created
    before any append. A failed upload kills recording for this run only.
    """

    def __init__(self, visitor_id=None, visitor_display_name=None, metadata=None):
        from wayfound import Session

        self._session = Session(
            wayfound_api_key=os.getenv("WAYFOUND_API_KEY"),
            agent_id=os.getenv("WAYFOUND_AGENT_ID", DEFAULT_AGENT_ID),
            visitor_id=visitor_id,
            visitor_display_name=visitor_display_name,
            metadata=metadata,
        )
        self._lock = threading.Lock()
        self._created = False
        self._dead = False

    def record(self, messages: list) -> None:
        """Queue a batch of messages for upload. Never blocks, never raises."""
        if not messages or self._dead:
            return
        threading.Thread(
            target=self._send, args=(list(messages),), daemon=True
        ).start()

    def _send(self, messages: list) -> None:
        with self._lock:  # serialize: create first, then appends in order
            if self._dead:
                return
            try:
                if self._created:
                    self._session.append_to_session(messages=messages, is_async=True)
                else:
                    self._session.create(messages=messages, is_async=True)
                    self._created = True
            except Exception as e:
                self._dead = True
                print(f"Wayfound: upload failed ({e}) — supervision disabled for this run")


def start_recorder(result: dict, username: str = None):
    """
    Build a recorder for one finished orchestrator run, tagging the session
    with the plan metadata. Returns None when Wayfound is not configured —
    callers can simply `if recorder:` around every use.
    """
    if not wayfound_available():
        return None
    plan = result.get("plan") or {}
    metadata = {
        "app": "query_app",
        "intent": plan.get("intent") or "",
        "period": plan.get("period") or "",
        "response_type": result.get("response_type") or "",
        "agents": ", ".join(plan.get("agents") or []),
    }
    try:
        return WayfoundRecorder(
            visitor_id=username,
            visitor_display_name=username,
            metadata=metadata,
        )
    except Exception as e:
        print(f"Wayfound: recorder init failed ({e}) — run will not be supervised")
        return None


# ============================================================
# Message builders — translate result dicts into Wayfound events
# ============================================================

def _clip(text: str) -> str:
    text = str(text or "")
    if len(text) > _MAX_CONTENT_CHARS:
        return text[:_MAX_CONTENT_CHARS] + " …[truncated]"
    return text


def _norm_ts(ts: str) -> str:
    """Agent dicts stamp '+00:00'; Wayfound examples use 'Z'. Normalize."""
    return ts.replace("+00:00", "Z") if isinstance(ts, str) else ts


def _after(ts: str, since: str) -> bool:
    """True when ISO timestamp `ts` is at or after `since` (both UTC)."""
    try:
        return datetime.fromisoformat(ts) >= datetime.fromisoformat(since)
    except (ValueError, TypeError):
        return True  # unparseable → keep the message rather than drop it


def _msg(event_type, content, timestamp, label=None, description=None, extra=None):
    attributes = {"content": _clip(content)}
    for key, value in (extra or {}).items():
        if value not in (None, "", []):
            attributes[key] = value
    message = {
        "timestamp": _norm_ts(timestamp),
        "event_type": event_type,
        "attributes": attributes,
    }
    if label:
        message["label"] = label
    if description:
        message["description"] = description
    return message


def _section_content(key: str, d: dict) -> str:
    """What the user actually saw for this section, as one text block."""
    if key == "newsletter":
        return d.get("newsletter", "")
    if key == "stress_test":
        if d.get("error"):
            return f"Stress test failed: {d['error']}"
        return f"{d.get('tables', '')}\n\n{d.get('narrative', '')}".strip()
    if key == "risk" and d.get("metrics"):
        return f"PORTFOLIO METRICS:\n{d['metrics']}\n\n{d.get('analysis', '')}"
    return d.get("analysis", "")


def _section_messages(result: dict, since: str = None, fallback_ts: str = None) -> list:
    """
    One assistant_message per agent section present in the result, ordered by
    each agent's own completion timestamp (preserves the parallel-run
    timeline). `since` filters to sections refreshed after that instant —
    used to record only what a revision actually re-ran.
    """
    fallback_ts = fallback_ts or utc_now_iso()
    collected = []
    for key in _SECTIONS:
        d = result.get(key)
        if not isinstance(d, dict):
            continue
        ts = d.get("timestamp") or fallback_ts
        if since and not _after(ts, since):
            continue
        critique_log = d.get("critique_log") or []
        extra = {
            "agent": d.get("agent"),
            "period": d.get("period"),
            "sources": d.get("sources"),
            "data_warnings": d.get("data_warnings"),
            "scenarios_used": d.get("scenarios_used"),
            "critique_rounds": len(critique_log) or None,
            "self_critique": _clip("\n---\n".join(critique_log))[:2000] or None,
        }
        collected.append((ts, _msg(
            "assistant_message",
            _section_content(key, d),
            ts,
            label=key,
            description=_SECTION_LABELS.get(key),
            extra=extra,
        )))
    collected.sort(key=lambda pair: pair[0])
    return [message for _, message in collected]


def messages_for_run(query: str, result: dict, started_at: str) -> list:
    """Full initial run: user query → planner decision → each agent output."""
    messages = [_msg("user_message", query, started_at)]

    plan = result.get("plan")
    if plan:
        content = (
            f"Routing decision: intent={plan.get('intent')}, "
            f"agents=[{', '.join(plan.get('agents') or [])}], "
            f"response_type={plan.get('response_type')}, "
            f"period={plan.get('period')}. "
            f"Reasoning: {plan.get('reasoning', '')}"
        )
        messages.append(_msg(
            "assistant_message",
            content,
            result.get("timestamp") or started_at,
            label="planner",
            description="Orchestrator intent classification",
            extra={"intent": plan.get("intent")},
        ))

    messages.extend(_section_messages(result, fallback_ts=started_at))
    return messages


def messages_for_revision(feedback: str, updated_result: dict, started_at: str,
                          section: str = None, via: str = "section_revise") -> list:
    """A per-section revision: the feedback plus only the re-run sections."""
    messages = [_msg(
        "user_message",
        feedback,
        started_at,
        label="revision_request",
        description=f"Revision requested via {via}",
        extra={"section": section, "via": via},
    )]
    messages.extend(_section_messages(updated_result, since=started_at))
    return messages


def messages_for_validation_plan(complaint: str, plan: dict, asked_at: str) -> list:
    """The validation agent's diagnosis step (before the user confirms)."""
    return [
        _msg(
            "user_message",
            complaint,
            asked_at,
            label="revision_request",
            description="Complaint submitted to validation_agent",
            extra={"via": "validation_agent"},
        ),
        _msg(
            "assistant_message",
            plan.get("summary", ""),
            utc_now_iso(),
            label="validation_planner",
            description="validation_agent revision plan (pending user confirm)",
            extra={
                "owner": plan.get("owner"),
                "diagnosis": plan.get("diagnosis"),
                "diagnosis_type": plan.get("diagnosis_type"),
                "materiality": plan.get("materiality"),
                "dirty_set": plan.get("dirty_set"),
                "targeted_query": plan.get("targeted_query"),
            },
        ),
    ]


def messages_for_validation_execution(updated_result: dict, started_at: str) -> list:
    """A confirmed validation plan: the re-run sections plus the agent's own
    explanation of what it changed."""
    messages = _section_messages(updated_result, since=started_at)
    explanation = updated_result.get("validation_explanation")
    if explanation:
        messages.append(_msg(
            "assistant_message",
            explanation,
            utc_now_iso(),
            label="validation_agent",
            description="validation_agent explanation of the applied revision",
        ))
    return messages
