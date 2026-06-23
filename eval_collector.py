"""
eval_collector.py — Example output collector for evaluation and future fine-tuning.

WHAT THIS DOES
--------------
1. Saves every agent pipeline run as a structured example in MongoDB
   (collection: eval_examples)
2. Provides a CLI to rate saved examples (supervisor feedback)
3. Exports rated examples as JSONL — ready for fine-tuning or few-shot injection
4. Runs a predefined test suite to detect regressions across pipeline versions

WHY EACH FORMAT
---------------
Newsletter / Analysis examples
    Used for: regression testing, few-shot prompt injection, style fine-tuning later.
    You need: the full input (query, period, PnL data snapshot), the output, and a
    quality rating. Rating dimensions: accuracy, insight, tone, format (1–5 each).

Trade Ranking examples  ← FUTURE, schema ready
    Used for: training a small classifier model (Llama, etc.) to pre-filter trade
    ideas before they reach the expensive model.
    You need: trade description + market regime → supervisor verdict (match: Y/N,
    rank: 1–5, notes). Even 50 labeled examples is enough to start.
    WHO LABELS: the portfolio manager / supervisor. This cannot be automated.

Market Regime Detection examples  ← FUTURE, schema ready
    Used for: classifying the current macro regime from market data, then mapping
    to a trade universe. Supervisor labels the regime for each period.
    WHO LABELS: the PM. Labels are: regime name, dominant force, expected duration.

COLLECTIONS
-----------
eval_examples       All saved examples (any category)
eval_ratings        Human ratings stored separately so examples can be re-rated

USAGE
-----
# Save a pipeline run manually (or call from query_app.py):
    from eval_collector import EvalCollector
    collector = EvalCollector()
    collector.save_example(result, category="newsletter")

# Rate examples via CLI:
    python eval_collector.py rate

# Export to JSONL for fine-tuning:
    python eval_collector.py export --category newsletter --min-rating 4

# Run the test suite (regression check):
    python eval_collector.py test

# Inspect saved examples:
    python eval_collector.py list
    python eval_collector.py show <example_id>
"""

import os
import sys
import json
import uuid
import asyncio
import argparse
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


# ============================================================
# Schema helpers
# ============================================================

def _make_example(
    result: dict,
    category: str,
    query: str | None = None,
) -> dict:
    """
    Build a structured example document from an agent pipeline result.

    category options:
        newsletter          Full newsletter pipeline output
        risk_analysis       Risk-only or full_brief output
        market_analysis     Market context only
        performance         Performance analysis only
        stress_test         Stress test output
        trade_eval          Trade ranking (future — fill manually)
        regime_detection    Market regime label (future — fill manually)
    """
    q = query or result.get("question", "")
    plan = result.get("plan", {})

    # ── Extract the actual text outputs ─────────────────────
    outputs: dict[str, Any] = {}

    if result.get("newsletter"):
        outputs["newsletter"] = result["newsletter"].get("newsletter", "")

    if result.get("performance"):
        outputs["performance_analysis"] = result["performance"].get("analysis", "")
        outputs["pnl_summary"] = result["performance"].get("pnl_summary")

    if result.get("market"):
        outputs["market_context"] = result["market"].get("analysis", "")

    if result.get("risk"):
        outputs["risk_analysis"] = result["risk"].get("analysis", "")
        outputs["risk_metrics_table"] = result["risk"].get("metrics", "")

    if result.get("stress_test"):
        st = result["stress_test"]
        outputs["stress_test_tables"]    = st.get("tables", "")
        outputs["stress_test_narrative"] = st.get("narrative", "")
        outputs["stress_test_scenarios"] = st.get("scenarios_used", [])

    # ── Word counts for quick quality checks ────────────────
    word_counts = {k: len(v.split()) if isinstance(v, str) else None
                   for k, v in outputs.items() if v}

    return {
        "_id":         str(uuid.uuid4()),
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "schema_ver":  "1.0",
        "category":    category,

        # ── Input ────────────────────────────────────────────
        "input": {
            "query":         q,
            "period":        plan.get("period") or result.get("performance", {}).get("period"),
            "response_type": result.get("response_type", ""),
            "intent":        plan.get("intent", ""),
            "agents_run":    plan.get("agents", []),
        },

        # ── Output ───────────────────────────────────────────
        "output":      outputs,
        "word_counts": word_counts,

        # ── Human rating (filled via `rate` command) ─────────
        "human_rating": {
            "overall":   None,   # 1–5  overall quality
            "accuracy":  None,   # 1–5  are the facts and figures correct?
            "insight":   None,   # 1–5  is the analysis genuinely useful?
            "tone":      None,   # 1–5  matches the fund's voice?
            "format":    None,   # 1–5  structure and readability
            "rated_by":  None,
            "rated_at":  None,
            "notes":     "",
        },

        # ── Trade eval fields (future — filled manually) ─────
        # Uncomment and populate when building the trade ranking dataset.
        # "trade_eval": {
        #     "trade_description": "",
        #     "market_regime":     "",
        #     "match_verdict":     None,   # true / false
        #     "rank":              None,   # 1–5 (5 = best fit)
        #     "rank_rationale":    "",
        #     "ranked_by":         None,
        #     "ranked_at":         None,
        # },

        # ── Regime detection fields (future — filled manually) ─
        # "regime": {
        #     "period":           "",     # e.g. "2026-02"
        #     "label":            "",     # e.g. "US rate shock"
        #     "dominant_force":   "",     # e.g. "real rates rising"
        #     "sub_regime":       "",     # e.g. "early hiking cycle"
        #     "expected_horizon": "",     # e.g. "3–6 months"
        #     "labeled_by":       None,
        #     "labeled_at":       None,
        # },

        # ── Revision history (if this example was revised) ───
        "revisions": result.get("revisions", []),
    }


# ============================================================
# EvalCollector class
# ============================================================

class EvalCollector:
    COLLECTION = "eval_examples"

    def __init__(self):
        self._client = MongoClient(os.getenv("MONGO_URI_USER"))
        self._db     = self._client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]
        self._col    = self._db[self.COLLECTION]

    # ── Write ────────────────────────────────────────────────

    def save_example(self, result: dict, category: str = "newsletter") -> str:
        """
        Save a pipeline result as a structured eval example.
        Returns the example _id.

        Call this from query_app.py after every successful run:
            from eval_collector import EvalCollector
            EvalCollector().save_example(result, category=result.get('response_type', 'newsletter'))
        """
        doc = _make_example(result, category)
        self._col.insert_one(doc)
        print(f"  ✓ Saved eval example [{doc['_id'][:8]}] category={category}")
        return doc["_id"]

    # ── Read ─────────────────────────────────────────────────

    def list_examples(
        self,
        category: str | None = None,
        rated_only: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        filt: dict = {}
        if category:
            filt["category"] = category
        if rated_only:
            filt["human_rating.overall"] = {"$ne": None}
        docs = list(
            self._col.find(filt, {"_id": 1, "created_at": 1, "category": 1,
                                   "input": 1, "human_rating.overall": 1,
                                   "word_counts": 1})
            .sort("created_at", -1)
            .limit(limit)
        )
        return docs

    def get_example(self, example_id: str) -> dict | None:
        return self._col.find_one({"_id": {"$regex": f"^{example_id}"}})

    # ── Rate ─────────────────────────────────────────────────

    def rate_example(
        self,
        example_id: str,
        overall: int,
        accuracy: int,
        insight: int,
        tone: int,
        format_: int,
        rated_by: str,
        notes: str = "",
    ) -> bool:
        """
        Store a human rating for an example.
        Scores are 1–5.
        """
        for score in [overall, accuracy, insight, tone, format_]:
            if not (1 <= score <= 5):
                raise ValueError(f"All scores must be 1–5, got {score}")

        result = self._col.update_one(
            {"_id": {"$regex": f"^{example_id}"}},
            {"$set": {
                "human_rating.overall":  overall,
                "human_rating.accuracy": accuracy,
                "human_rating.insight":  insight,
                "human_rating.tone":     tone,
                "human_rating.format":   format_,
                "human_rating.rated_by": rated_by,
                "human_rating.rated_at": datetime.now(timezone.utc).isoformat(),
                "human_rating.notes":    notes,
            }}
        )
        return result.modified_count == 1

    # ── Export ───────────────────────────────────────────────

    def export_jsonl(
        self,
        path: str,
        category: str | None = None,
        min_rating: int = 4,
        format_: str = "chat",
    ) -> int:
        """
        Export rated examples as JSONL.

        format_ options:
            "raw"   — full example doc (for analysis / evals)
            "chat"  — Anthropic fine-tuning format:
                      {"messages": [{"role":"user",...}, {"role":"assistant",...}]}

        Returns number of examples exported.
        """
        filt: dict = {"human_rating.overall": {"$gte": min_rating}}
        if category:
            filt["category"] = category

        docs = list(self._col.find(filt))
        if not docs:
            print(f"  No examples found with rating >= {min_rating}")
            return 0

        with open(path, "w") as f:
            for doc in docs:
                if format_ == "chat":
                    row = _to_chat_format(doc)
                    if row:
                        f.write(json.dumps(row) + "\n")
                else:
                    # Remove MongoDB _id field for clean export
                    doc.pop("_id", None)
                    f.write(json.dumps(doc, default=str) + "\n")

        print(f"  ✓ Exported {len(docs)} examples to {path}")
        return len(docs)

    # ── Stats ────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a summary of the eval dataset."""
        pipeline = [
            {"$group": {
                "_id":           "$category",
                "count":         {"$sum": 1},
                "rated":         {"$sum": {"$cond": [{"$ne": ["$human_rating.overall", None]}, 1, 0]}},
                "avg_rating":    {"$avg": "$human_rating.overall"},
            }}
        ]
        return {doc["_id"]: doc for doc in self._col.aggregate(pipeline)}


def _to_chat_format(doc: dict) -> dict | None:
    """
    Convert a saved example to Anthropic fine-tuning chat format.
    Currently supported for newsletter and risk_analysis categories.
    """
    category = doc.get("category", "")
    inp      = doc.get("input", {})
    out      = doc.get("output", {})

    user_content = inp.get("query", "")
    if not user_content:
        return None

    if category == "newsletter" and out.get("newsletter"):
        assistant_content = out["newsletter"]
    elif category == "risk_analysis" and out.get("risk_analysis"):
        assistant_content = out["risk_analysis"]
    elif category == "market_analysis" and out.get("market_context"):
        assistant_content = out["market_context"]
    elif category == "performance" and out.get("performance_analysis"):
        assistant_content = out["performance_analysis"]
    else:
        return None

    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ============================================================
# Pre-defined test suite
# ============================================================

# Each test case: query + expected response_type + minimum word count thresholds.
# Add more as you discover important edge cases.

TEST_SUITE = [
    {
        "id":             "newsletter_feb26",
        "query":          "Write a February 2026 investor newsletter.",
        "expected_intent":"newsletter",
        "expected_type":  "newsletter",
        "min_words": {
            "newsletter":         800,
            "performance_analysis": 200,
            "market_context":       150,
            "risk_analysis":        300,
        },
    },
    {
        "id":             "risk_feb26",
        "query":          "Give me a risk analysis for February 2026.",
        "expected_intent":"risk_analysis",
        "expected_type":  "analysis",
        "min_words": {
            "risk_analysis":      300,
            "performance_analysis": 100,
        },
    },
    {
        "id":             "market_only_feb26",
        "query":          "What happened in macro markets in February 2026?",
        "expected_intent":"market_analysis",
        "expected_type":  "analysis",
        "min_words": {
            "market_context": 150,
        },
    },
    {
        "id":             "performance_only_feb26",
        "query":          "How did the portfolio perform in February 2026?",
        "expected_intent":"performance",
        "expected_type":  "analysis",
        "min_words": {
            "performance_analysis": 200,
        },
    },
    {
        "id":             "full_brief_feb26",
        "query":          "Give me a full analysis brief for February 2026.",
        "expected_intent":"full_brief",
        "expected_type":  "analysis",
        "min_words": {
            "performance_analysis": 200,
            "market_context":       150,
            "risk_analysis":        300,
        },
    },
    {
        "id":             "stress_test_feb26",
        "query":          "Stress test the February 2026 portfolio against a rate shock.",
        "expected_intent":"stress_test",
        "expected_type":  "analysis",
        "min_words": {
            "stress_test_narrative": 300,
        },
    },
]


async def _run_test_case(orchestrator, tc: dict) -> dict:
    """Run a single test case and return a result report."""
    from multiagent import _extract_period

    start = datetime.now(timezone.utc)
    result = await orchestrator.run(tc["query"])
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()

    plan    = result.get("plan", {})
    outputs = {}
    if result.get("newsletter"):
        outputs["newsletter"] = result["newsletter"].get("newsletter", "")
    if result.get("performance"):
        outputs["performance_analysis"] = result["performance"].get("analysis", "")
    if result.get("market"):
        outputs["market_context"] = result["market"].get("analysis", "")
    if result.get("risk"):
        outputs["risk_analysis"] = result["risk"].get("analysis", "")
    if result.get("stress_test"):
        st = result["stress_test"]
        outputs["stress_test_narrative"] = st.get("narrative", "")

    checks = []

    # Check 1: intent routing
    actual_intent = plan.get("intent", "")
    intent_ok = actual_intent == tc["expected_intent"]
    checks.append({
        "check":    "intent",
        "expected": tc["expected_intent"],
        "actual":   actual_intent,
        "pass":     intent_ok,
    })

    # Check 2: response_type
    actual_type = result.get("response_type", "")
    type_ok = actual_type == tc["expected_type"]
    checks.append({
        "check":    "response_type",
        "expected": tc["expected_type"],
        "actual":   actual_type,
        "pass":     type_ok,
    })

    # Check 3: minimum word counts
    for key, min_words in tc.get("min_words", {}).items():
        text  = outputs.get(key, "")
        words = len(text.split()) if text else 0
        checks.append({
            "check":    f"min_words:{key}",
            "expected": min_words,
            "actual":   words,
            "pass":     words >= min_words,
        })

    all_pass = all(c["pass"] for c in checks)

    return {
        "test_id":   tc["id"],
        "query":     tc["query"],
        "elapsed_s": round(elapsed, 1),
        "pass":      all_pass,
        "checks":    checks,
        "result":    result,
    }


async def run_test_suite(orchestrator, save: bool = True) -> list[dict]:
    """
    Run all TEST_SUITE cases, print a report, optionally save outputs to eval_examples.
    Returns list of test reports.
    """
    collector = EvalCollector() if save else None
    reports   = []

    print(f"\n{'=' * 60}")
    print(f"EVAL TEST SUITE  —  {len(TEST_SUITE)} test cases")
    print(f"{'=' * 60}\n")

    for tc in TEST_SUITE:
        print(f"  Running: [{tc['id']}] {tc['query'][:60]}...")
        try:
            report = await _run_test_case(orchestrator, tc)
        except Exception as e:
            report = {
                "test_id": tc["id"],
                "query":   tc["query"],
                "pass":    False,
                "error":   str(e),
                "checks":  [],
            }

        reports.append(report)

        # Print check results
        status = "✓ PASS" if report.get("pass") else "✗ FAIL"
        elapsed = report.get("elapsed_s", "?")
        print(f"  {status}  [{elapsed}s]")
        for chk in report.get("checks", []):
            icon = "  ✓" if chk["pass"] else "  ✗"
            print(f"    {icon} {chk['check']}: expected={chk['expected']} actual={chk['actual']}")

        # Save to MongoDB
        if save and "result" in report:
            plan_intent = report["result"].get("plan", {}).get("intent", "unknown")
            collector.save_example(
                report["result"],
                category=plan_intent,
            )
        print()

    # Summary
    passed = sum(1 for r in reports if r.get("pass"))
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{len(reports)} passed")
    print(f"{'=' * 60}\n")

    return reports


# ============================================================
# CLI
# ============================================================

def _truncate(s: str | None, n: int = 80) -> str:
    if not s:
        return "(empty)"
    s = s.replace("\n", " ")
    return s[:n] + "..." if len(s) > n else s


def cmd_list(args):
    collector = EvalCollector()
    docs = collector.list_examples(
        category=getattr(args, "category", None),
        rated_only=getattr(args, "rated", False),
        limit=getattr(args, "limit", 50),
    )
    if not docs:
        print("No examples found.")
        return

    print(f"\n{'ID':10}  {'Category':18}  {'Rating':7}  {'Period':8}  Query")
    print("-" * 90)
    for d in docs:
        eid     = d["_id"][:8]
        cat     = d.get("category", "?")[:18]
        rating  = d.get("human_rating", {}).get("overall") or "—"
        period  = (d.get("input") or {}).get("period") or "?"
        query   = _truncate((d.get("input") or {}).get("query", ""), 50)
        print(f"{eid:10}  {cat:18}  {str(rating):7}  {period:8}  {query}")
    print(f"\n{len(docs)} example(s) shown.\n")


def cmd_show(args):
    collector = EvalCollector()
    doc = collector.get_example(args.id)
    if not doc:
        print(f"No example found matching id prefix '{args.id}'")
        return

    print(f"\n{'=' * 60}")
    print(f"ID:       {doc['_id']}")
    print(f"Category: {doc['category']}")
    print(f"Created:  {doc['created_at']}")
    inp = doc.get("input", {})
    print(f"\nINPUT")
    print(f"  Query:   {inp.get('query', '')}")
    print(f"  Period:  {inp.get('period', '?')}")
    print(f"  Intent:  {inp.get('intent', '?')}")
    print(f"  Agents:  {inp.get('agents_run', [])}")

    wc = doc.get("word_counts", {})
    print(f"\nOUTPUT (word counts)")
    for k, v in wc.items():
        if v:
            print(f"  {k}: {v} words")

    rating = doc.get("human_rating", {})
    print(f"\nRATING")
    if rating.get("overall") is not None:
        print(f"  Overall:  {rating['overall']}/5  by {rating.get('rated_by', '?')}")
        print(f"  Accuracy: {rating['accuracy']}/5  Insight: {rating['insight']}/5  "
              f"Tone: {rating['tone']}/5  Format: {rating['format']}/5")
        if rating.get("notes"):
            print(f"  Notes:    {rating['notes']}")
    else:
        print("  Not yet rated.")

    if args.full:
        out = doc.get("output", {})
        for key, val in out.items():
            if val and isinstance(val, str):
                print(f"\n── {key.upper()} ──")
                print(val[:2000] + ("..." if len(val) > 2000 else ""))
    print()


def cmd_rate(args):
    """Interactive CLI rating session."""
    collector = EvalCollector()

    # If a specific ID was given, rate just that one
    if hasattr(args, "id") and args.id:
        docs = [collector.get_example(args.id)]
        docs = [d for d in docs if d]
    else:
        # Get unrated examples
        filt = {"human_rating.overall": None}
        if hasattr(args, "category") and args.category:
            filt["category"] = args.category
        docs = list(
            collector._col.find(filt, {"_id": 1, "category": 1, "input": 1, "output": 1,
                                        "word_counts": 1})
            .sort("created_at", -1)
            .limit(getattr(args, "limit", 10))
        )

    if not docs:
        print("No unrated examples found.")
        return

    rated_by = input("Your name / initials: ").strip() or "anon"
    print(f"\nRating {len(docs)} example(s). Press Ctrl-C to stop.\n")

    for doc in docs:
        eid    = doc["_id"][:8]
        cat    = doc.get("category", "?")
        query  = (doc.get("input") or {}).get("query", "")
        period = (doc.get("input") or {}).get("period", "?")

        print(f"\n{'─' * 60}")
        print(f"  ID: {eid}  |  Category: {cat}  |  Period: {period}")
        print(f"  Query: {_truncate(query, 100)}")
        wc = doc.get("word_counts", {})
        if wc:
            print(f"  Outputs: {', '.join(f'{k}={v}w' for k, v in wc.items() if v)}")

        # Show a short preview of the main output
        out = doc.get("output", {})
        preview_key = next(
            (k for k in ["newsletter", "risk_analysis", "market_context",
                          "performance_analysis", "stress_test_narrative"] if out.get(k)),
            None
        )
        if preview_key:
            print(f"\n  Preview ({preview_key}):")
            print("  " + _truncate(out[preview_key], 300))

        print()

        try:
            def get_score(prompt: str) -> int:
                while True:
                    val = input(f"  {prompt} (1–5): ").strip()
                    if val.isdigit() and 1 <= int(val) <= 5:
                        return int(val)
                    print("    Please enter a number between 1 and 5.")

            overall  = get_score("Overall quality      ")
            accuracy = get_score("Accuracy (facts)     ")
            insight  = get_score("Insight (useful?)    ")
            tone     = get_score("Tone (fund's voice?) ")
            fmt      = get_score("Format (structure)   ")
            notes    = input("  Notes (optional): ").strip()

            ok = collector.rate_example(
                doc["_id"], overall, accuracy, insight, tone, fmt,
                rated_by=rated_by, notes=notes,
            )
            print(f"  ✓ Saved." if ok else "  ✗ Not found — skipping.")

        except KeyboardInterrupt:
            print("\n\nRating session ended.")
            break

    print(f"\nDone. Run `python eval_collector.py list --rated` to see rated examples.\n")


def cmd_export(args):
    collector = EvalCollector()
    path     = getattr(args, "output", "eval_export.jsonl")
    category = getattr(args, "category", None)
    min_r    = getattr(args, "min_rating", 4)
    fmt      = getattr(args, "format", "chat")
    n = collector.export_jsonl(path, category=category, min_rating=min_r, format_=fmt)
    if n:
        print(f"  Exported {n} examples → {path}")


def cmd_stats(args):
    collector = EvalCollector()
    stats = collector.stats()
    if not stats:
        print("No examples in collection yet.")
        return
    print(f"\n{'Category':20}  {'Total':6}  {'Rated':6}  {'Avg Rating':10}")
    print("-" * 50)
    for cat, s in sorted(stats.items()):
        avg = f"{s['avg_rating']:.1f}" if s.get("avg_rating") else "—"
        print(f"  {cat:20}  {s['count']:6}  {s['rated']:6}  {avg:10}")
    print()


def cmd_test(args):
    from multiagent import build_agent_system
    orchestrator = build_agent_system()
    save = not getattr(args, "no_save", False)
    asyncio.run(run_test_suite(orchestrator, save=save))


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Eval collector — save, rate, and export pipeline examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  list     Show saved examples
  show     Show a single example in detail
  rate     Interactively rate unrated examples
  export   Export rated examples to JSONL (for fine-tuning)
  stats    Show dataset statistics
  test     Run the predefined test suite
        """
    )
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="List saved examples")
    p_list.add_argument("--category", "-c", help="Filter by category")
    p_list.add_argument("--rated",    "-r", action="store_true", help="Only show rated")
    p_list.add_argument("--limit",    "-n", type=int, default=50)

    # show
    p_show = sub.add_parser("show", help="Show a single example")
    p_show.add_argument("id", help="Example ID prefix (first 8 chars)")
    p_show.add_argument("--full", "-f", action="store_true", help="Show full output text")

    # rate
    p_rate = sub.add_parser("rate", help="Interactively rate examples")
    p_rate.add_argument("--id",       help="Rate a specific example ID")
    p_rate.add_argument("--category", "-c", help="Filter by category")
    p_rate.add_argument("--limit",    "-n", type=int, default=10)

    # export
    p_exp = sub.add_parser("export", help="Export to JSONL")
    p_exp.add_argument("--output",     "-o", default="eval_export.jsonl")
    p_exp.add_argument("--category",   "-c", help="Filter by category")
    p_exp.add_argument("--min-rating", "-m", type=int, default=4,
                       dest="min_rating", help="Minimum overall rating (default 4)")
    p_exp.add_argument("--format",     "-f", choices=["chat", "raw"], default="chat",
                       help="chat = Anthropic fine-tuning format, raw = full doc")

    # stats
    sub.add_parser("stats", help="Dataset statistics")

    # test
    p_test = sub.add_parser("test", help="Run predefined test suite")
    p_test.add_argument("--no-save", action="store_true",
                        help="Don't save test outputs to MongoDB")

    args = parser.parse_args()

    dispatch = {
        "list":   cmd_list,
        "show":   cmd_show,
        "rate":   cmd_rate,
        "export": cmd_export,
        "stats":  cmd_stats,
        "test":   cmd_test,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
