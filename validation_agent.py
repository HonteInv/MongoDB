"""
validation_agent.py — interpreting revision layer.

Takes a finished result + a free-text complaint, works out what actually needs
to change, and re-runs only the responsible agent(s) in dependency order.

Two-phase by design so the UI can confirm before anything runs:
    plan_revision(complaint, result)   → a PLAN, runs no agents
    execute_revision(plan, result)     → runs the plan, returns updated result

Engine only — all UI lives in query_app.py. multiagent.py is untouched; this
class takes an already-built OrchestratorAgent and drives its public agents.
"""

import os
import re
import json
import asyncio
from datetime import datetime, timezone

from pymongo import MongoClient

from helper import get_latest_opus_model


class ValidationAgent:
    # Which downstream sections read a given section's output.
    DEPENDENTS = {
        "market":      ["risk", "newsletter"],
        "performance": ["risk", "newsletter"],
        "weekly":      ["newsletter"],
        "risk":        ["newsletter"],
        "newsletter":  [],
    }

    # Canonical run order — cascade always replays in this order.
    PIPELINE_ORDER = ["market", "performance", "weekly", "risk", "newsletter"]

    VALID_OWNERS = set(PIPELINE_ORDER)

    def __init__(self, orchestrator, anthropic_client, model: str = None):
        self.orch = orchestrator
        self.client = anthropic_client
        self.model = model or os.getenv("CLAUDE_MODEL") or get_latest_opus_model(anthropic_client)

        # Dormant DB handle — unused in Tier 2, here so Tier 3 (real numeric
        # verification against pnl_summary / pnl_table) slots in without re-plumbing.
        self._mongo_client = MongoClient(os.getenv("MONGO_URI_USER"))

    def _db(self):
        return self._mongo_client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]

    # ----------------------------------------------------------
    # PHASE A — plan (runs no agents)
    # ----------------------------------------------------------

    async def plan_revision(self, complaint: str, current_result: dict) -> dict:
        """
        Interpret the complaint and return a plan. Runs no agents.

        Plan shape:
        {
          "complaint":       original text,
          "owner":           which section owns the complaint,
          "diagnosis":       short explanation,
          "diagnosis_type":  "data" | "generation",
          "targeted_query":  rewritten instruction for the owner agent,
          "materiality":     "material" | "cosmetic",
          "dirty_set":       ordered list of sections to re-run,
          "summary":         plain-English description for the confirm step,
        }
        """
        present = [s for s in self.PIPELINE_ORDER if s in current_result]

        system_prompt = (
            "You are a revision router for a portfolio analysis system. "
            "A user has a finished multi-section report and a complaint about it. "
            "Decide which ONE section the complaint is really about, whether it is a "
            "data problem or a generation/wording problem, and rewrite the complaint "
            "as a specific instruction for that section's agent.\n\n"
            "Sections:\n"
            "  market       — macro market context (rates, FX, equities, central banks)\n"
            "  performance  — portfolio P&L, returns, attribution (numbers live here)\n"
            "  weekly       — daily/weekly market data trends\n"
            "  risk         — DV01, notional, concentration, stress framing\n"
            "  newsletter   — the written investor letter (tone, structure, prose)\n\n"
            f"Sections present in this report: {present}\n\n"
            "diagnosis_type:\n"
            "  data       — a number/fact appears wrong (lives in the source data)\n"
            "  generation — the data is fine but the model misread, mis-worded, or mis-structured it\n\n"
            "materiality:\n"
            "  material   — the change alters a number or fact that other sections depend on\n"
            "  cosmetic   — tone/structure/wording only; downstream sections need not change\n\n"
            "Return ONLY valid JSON, no markdown fences:\n"
            "{\n"
            '  "owner": "performance",\n'
            '  "diagnosis": "one sentence",\n'
            '  "diagnosis_type": "data|generation",\n'
            '  "targeted_query": "specific instruction for the owner agent",\n'
            '  "materiality": "material|cosmetic"\n'
            "}"
        )

        user_prompt = f"COMPLAINT:\n{complaint}\n\nDecide the owner, diagnosis, rewrite, and materiality."

        owner = None
        parsed = {}
        try:
            resp = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=400,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                owner = parsed.get("owner")
        except Exception as e:
            print(f"  ValidationAgent.plan_revision failed ({e}) — falling back to newsletter owner")

        # Fallback if the router failed or returned something unusable
        if owner not in self.VALID_OWNERS or owner not in present:
            owner = "newsletter" if "newsletter" in present else (present[-1] if present else "performance")
            parsed.setdefault("diagnosis", "Could not classify automatically — defaulting to the main output.")
            parsed.setdefault("diagnosis_type", "generation")
            parsed.setdefault("targeted_query", complaint)
            parsed.setdefault("materiality", "material")

        materiality = parsed.get("materiality", "material")

        # Compute the dirty set deterministically (don't trust the LLM with ordering).
        dirty = [owner]
        if materiality == "material":
            dirty += [d for d in self.DEPENDENTS.get(owner, []) if d in present]
        dirty = [s for s in self.PIPELINE_ORDER if s in dirty]  # canonical order, deduped

        plan = {
            "complaint":      complaint,
            "owner":          owner,
            "diagnosis":      parsed.get("diagnosis", ""),
            "diagnosis_type": parsed.get("diagnosis_type", "generation"),
            "targeted_query": parsed.get("targeted_query", complaint),
            "materiality":    materiality,
            "dirty_set":      dirty,
            "summary":        self._build_summary(owner, dirty, parsed, present),
        }
        return plan

    def _build_summary(self, owner, dirty, parsed, present) -> str:
        """Deterministic plain-English plan for the confirm step."""
        label = {
            "market": "Market Context", "performance": "Portfolio Performance",
            "weekly": "Daily Market Data", "risk": "Risk Analysis",
            "newsletter": "Newsletter",
        }
        downstream = [s for s in dirty if s != owner]
        untouched = [s for s in present if s not in dirty]

        dtype = parsed.get("diagnosis_type", "generation")
        kind = "a data issue" if dtype == "data" else "a wording / generation issue"

        lines = [f"This looks like {kind} in the **{label.get(owner, owner)}** section."]
        if parsed.get("diagnosis"):
            lines.append(f"_{parsed['diagnosis']}_")
        lines.append(f"• Re-run **{label.get(owner, owner)}**")
        for d in downstream:
            lines.append(f"• Refresh **{label.get(d, d)}** to stay consistent")
        if untouched:
            lines.append(f"• Leave untouched: {', '.join(label.get(s, s) for s in untouched)}")
        return "\n\n".join(lines)

    # ----------------------------------------------------------
    # PHASE B — execute (runs the dirty set in order)
    # ----------------------------------------------------------

    async def execute_revision(self, plan: dict, current_result: dict) -> dict:
        """
        Run the dirty set in dependency order. Only the OWNER receives the
        targeted query as feedback; downstream sections re-run plain so they
        pick up the owner's refreshed output and stay consistent.
        """
        import copy
        result = copy.deepcopy(current_result)
        question = result.get("question", "")
        owner = plan["owner"]
        targeted = plan["targeted_query"]

        def fb(section):  # feedback only for the owning section
            return targeted if section == owner else ""

        for section in plan["dirty_set"]:
            if section == "market":
                result["market"] = await asyncio.to_thread(
                    self.orch.market.analyze, question, fb("market")
                )
            elif section == "performance":
                result["performance"] = await asyncio.to_thread(
                    self.orch.performance.analyze, question, fb("performance")
                )
            elif section == "weekly":
                # WeeklyMarketDataAgent.analyze takes no feedback param — refresh only
                result["weekly"] = await asyncio.to_thread(
                    self.orch.weekly.analyze, question
                )
            elif section == "risk":
                result["risk"] = await asyncio.to_thread(
                    self.orch.risk.analyze,
                    question,
                    result.get("market", {}).get("analysis", ""),
                    result.get("performance", {}).get("analysis", ""),
                    fb("risk"),
                )
            elif section == "newsletter":
                result["newsletter"] = await asyncio.to_thread(
                    self.orch.writer.write,
                    question,
                    result.get("market", {}).get("analysis", ""),
                    result.get("performance", {}).get("analysis", ""),
                    result.get("risk", {}).get("analysis", ""),
                    result.get("weekly", {}).get("analysis", ""),
                    result.get("performance", {}).get("pnl_summary"),
                    fb("newsletter"),
                )

        result["validation_explanation"] = self._build_explanation(plan)

        # Append to the revisions log (same shape multiagent.py uses)
        result.setdefault("revisions", []).append({
            "section":   plan["owner"],
            "feedback":  plan["complaint"],
            "via":       "validation_agent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return result

    def _build_explanation(self, plan: dict) -> str:
        label = {
            "market": "Market Context", "performance": "Portfolio Performance",
            "weekly": "Daily Market Data", "risk": "Risk Analysis",
            "newsletter": "Newsletter",
        }
        ran = ", ".join(label.get(s, s) for s in plan["dirty_set"])
        return (
            f"Re-ran: {ran}. "
            f"Interpreted your note as {plan['diagnosis_type']} issue in "
            f"{label.get(plan['owner'], plan['owner'])} → \"{plan['targeted_query']}\""
        )

    @staticmethod
    def version_label(plan: dict) -> str:
        """Concise summary of what a revision changed — used as the version-stack label."""
        label = {
            "market": "market", "performance": "performance", "weekly": "weekly",
            "risk": "risk", "newsletter": "newsletter",
        }
        return f"{plan['owner']} revision — {', '.join(label.get(s, s) for s in plan['dirty_set'])}"
