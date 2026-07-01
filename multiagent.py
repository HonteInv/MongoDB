"""
Multi-Agent
"""

import re
import time
import asyncio
import os
import calendar
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

from dotenv import load_dotenv
from anthropic import Anthropic

from helper import get_latest_opus_model

from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings


# ============================================================
# MongoDB Helper
# ============================================================

def get_vector_store(collection_name: str, index_name: str, embedding) -> MongoDBAtlasVectorSearch:
    """
    Returns a LangChain-compatible MongoDB vector store.
    Replaces FAISS.load_local() entirely.
    """
    client = MongoClient(os.getenv("MONGO_URI_USER"))
    collection = client[os.getenv("MONGO_DB_NAME", "portfolio_rag")][collection_name]

    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding,
        index_name=index_name,
        text_key="text",
        embedding_key="embedding",
    )


# ============================================================
# Base Agent
# ============================================================

class BaseAgent:
    def __init__(self, embedding, anthropic_client: Anthropic, context_k: int = 30, model: str = None):
        self.embedding = embedding
        self.client = anthropic_client
        self.context_k = context_k

        # Model resolved once at system startup and passed in — avoids repeated API calls.
        # Falls back to resolving here only if used standalone outside build_agent_system().
        self.model = model or os.getenv("CLAUDE_MODEL") or get_latest_opus_model(anthropic_client)
        self.max_tokens = int(os.getenv("CLAUDE_MAX_OUTPUT_TOKENS", "8192"))
        # temperature is deprecated for Opus 4.7+ — removed

    def _search(self, store: MongoDBAtlasVectorSearch, query: str, k: int = 20):
        """
        Replaces hybrid_search() + rerank().
        MongoDB Atlas Vector Search handles similarity directly.
        """
        return store.similarity_search(query, k=k)

    def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude with automatic retry on rate limit errors (429)."""
        wait = 15  # seconds before first retry
        for attempt in range(4):
            try:
                response = self.client.messages.create(
                    model = self.model,
                    max_tokens = self.max_tokens,
                    system = system_prompt,
                    messages = [{"role": "user", "content": user_prompt}],
                )
                return "".join(
                    block.text for block in response.content if hasattr(block, "text")
                ).strip()
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < 3:
                    print(f"Rate limit hit — waiting {wait}s before retry {attempt + 1}/3...")
                    time.sleep(wait)
                    wait *= 2  # exponential backoff: 15 → 30 → 60s
                else:
                    raise

    def _critique(self, draft: str, rubric: str) -> str:
        """
        Ask the model to critique a draft against a rubric.
        Returns a short critique — either 'PASS' or a list of specific issues.
        Uses the base model (haiku) to keep costs low.
        """
        response = self.client.messages.create(
            model=os.getenv("CLAUDE_MODEL") or get_latest_opus_model(self.client),
            max_tokens=512,
            system=(
                "You are a strict editorial reviewer. "
                "Evaluate the draft against the rubric provided. "
                "If the draft meets all criteria, respond with exactly: PASS\n"
                "If it fails any criteria, respond with a numbered list of specific issues only. "
                "Be concise — maximum 5 issues. Do not rewrite the draft."
            ),
            messages=[{"role": "user", "content": f"RUBRIC:\n{rubric}\n\nDRAFT:\n{draft}"}],
        )
        return "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()

    def _call_claude_with_critique(
        self,
        system_prompt: str,
        user_prompt: str,
        rubric: str,
        max_retries: int = 2,
    ) -> tuple[str, list[str]]:
        """
        Generate a response, critique it, and revise if needed.
        Returns (final_output, list_of_critique_rounds).
        Falls back to the original draft if retries are exhausted.

        max_retries=0 disables critique and behaves identically to _call_claude().
        """
        critique_log = []
        draft = self._call_claude(system_prompt, user_prompt)

        for attempt in range(max_retries):
            critique = self._critique(draft, rubric)
            critique_log.append(critique)

            if critique.strip().upper() == "PASS":
                break

            # Revise — feed the critique back into the same system prompt context
            revision_prompt = (
                f"{user_prompt}\n\n"
                f"---\n"
                f"Your previous draft had the following issues:\n{critique}\n\n"
                f"Please rewrite addressing each issue. Keep everything that was correct."
            )
            draft = self._call_claude(system_prompt, revision_prompt)

        return draft, critique_log


# ============================================================
# Market Series Agent (daily/weekly numeric time-series → "recent moves")
# ============================================================

class MarketSeriesAgent:
    """
    Grabs the daily/weekly market time-series for the query's period from the
    structured `market_series` collection and computes DETERMINISTIC changes
    (yields/swaps in bp, prices in %), labeling fixed-expiry contracts correctly.

    Output is a compact "recent market moves" block that MarketContextAgent folds
    into its analysis. Numbers here are computed, never model-generated — so the
    letter can cite them safely. Returns an empty block when there's no data for
    the period (honest, no fabrication).
    """

    def __init__(self, model: str = None):
        self.model = model  # unused today (pure computation) — kept for parity

    def summarize(self, question: str) -> Dict[str, Any]:
        import market_series as ms

        period = _extract_period(question)
        empty = {
            "agent": "MarketSeriesAgent", "moves_block": "", "period": period,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if not period:
            return empty

        yyyy, mm = int(period[:4]), int(period[5:7])
        last_day = calendar.monthrange(yyyy, mm)[1]
        start = f"{period}-01"
        end = f"{period}-{last_day:02d}"
        # Reach back ~7 days so the first move has a prior reference point.
        lookback = (datetime.fromisoformat(start) - timedelta(days=7)).strftime("%Y-%m-%d")

        try:
            rows = ms.get_range(lookback, end)
        except Exception as e:
            print(f"MarketSeriesAgent: market_series query failed ({e})")
            return empty
        if not rows:
            return empty

        # Prefer daily granularity when available; else weekly.
        freqs = {r["frequency"] for r in rows}
        freq = "daily" if "daily" in freqs else next(iter(freqs))
        rows = [r for r in rows if r["frequency"] == freq]
        rows.sort(key=lambda r: r["date"])

        # Change is measured across the requested month; fall back to the full
        # lookback window if the month itself has fewer than two observations.
        month_rows = [r for r in rows if start <= r["date"] <= end]
        window = month_rows if len(month_rows) >= 2 else rows
        if len(window) < 2:
            return empty

        first, last = window[0], window[-1]
        _, ticker_meta, _ = ms.load_ticker_map()

        lines = [f"RECENT MARKET MOVES (computed from market data, {freq}, "
                 f"{first['date']} → {last['date']}):"]
        for ticker in sorted(set(first["series"]) & set(last["series"])):
            a, b = first["series"][ticker], last["series"][ticker]
            meta = ticker_meta.get(ticker, {"label": ticker, "unit": "price", "type": "generic"})
            unit = meta["unit"]
            if unit == "percent":            # rate quoted in % → change in bp
                chg_s = f"{(b - a) * 100:+.0f}bp"
            elif unit == "bp":               # value already in bp → raw difference
                chg_s = f"{(b - a):+.1f}bp"
            else:                            # price/level → percent change
                chg_s = f"{((b - a) / a * 100):+.1f}%" if a else "n/a"
            tag = " [fixed-expiry contract]" if meta["type"] == "fixed_expiry" else ""
            lines.append(f"- {meta['label']}: {a:g} → {b:g} ({chg_s}){tag}")

        return {
            "agent": "MarketSeriesAgent",
            "moves_block": "\n".join(lines),
            "period": period,
            "frequency": freq,
            "as_of": last["date"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================
# Market Context Agent
# ============================================================

class MarketContextAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Daily-tagged narrative collection (date/section/topic pre-filtering)
        self.context_store = get_vector_store(
            "context_daily",
            "daily_index",
            self.embedding,
        )
        # Exact-number tables (structured, queried by date — not vectors)
        self._mongo = MongoClient(os.getenv("MONGO_URI_USER"))
        # Daily/weekly market time-series → "recent moves" (structured, deterministic)
        self.series = MarketSeriesAgent(model=getattr(self, "model", None))
        # CLAUDE_CONTEXT_MODEL env var takes priority; otherwise use the model
        # already resolved and passed in from build_agent_system().
        if os.getenv("CLAUDE_CONTEXT_MODEL"):
            self.model = os.getenv("CLAUDE_CONTEXT_MODEL")

    def _tables_col(self):
        return self._mongo[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["daily_table_data"]

    def _get_period_tables(self, period: str) -> tuple[str, list[str]]:
        """
        Exact month-end closes for the period from daily_table_data. Returns
        (formatted_block, data_warnings). The block is "" if no tables exist
        (e.g. before backfill). data_warnings lists any tables flagged incomplete
        or low-confidence, so the agent can disclose gaps to the user.
        """
        col = self._tables_col()
        days = sorted(col.distinct("report_day", {"report_month": period}))
        if not days:
            return "", []
        last = days[-1]  # month-end snapshot
        lines = [f"Exact market closes as of {last} (month-end):"]
        warnings: list[str] = []
        for doc in col.find({"report_day": last}, {"_id": 0}):
            q = doc.get("quality", {}) or {}
            flagged = q.get("incomplete") or (q.get("confidence") or 100) < 95 or q.get("empty_cells")
            note = "  [DATA NOTE: this table may be incomplete or uncertain]" if flagged else ""
            if flagged:
                warnings.append(
                    f"{doc['exhibit']} ({last}): possible missing/uncertain values "
                    f"(confidence {q.get('confidence')}%)"
                )
            lines.append(f"\n[{doc['exhibit']}]{note}")
            for row, cols in doc.get("table", {}).items():
                vals = ", ".join(f"{k}={'?' if v is None else v}" for k, v in cols.items())
                lines.append(f"  {row}: {vals}")
        return "\n".join(lines), warnings

    def _build_macro_query(self, question: str) -> str:
        """Reframe the question as a macro-focused vector search query."""
        period = _extract_period(question)
        if period:
            # Convert YYYY-MM to "Month YYYY" for readable query
            month_names = {v: k.capitalize() for k, v in _MONTH_MAP.items() if len(k) > 3}
            mm = period.split("-")[1]
            yyyy = period.split("-")[0]
            month_label = month_names.get(mm, mm)
            return (
                f"{month_label} {yyyy} macro market events rates FX equities "
                f"central bank Fed monetary policy inflation commodities"
            )
        return "macro market events rates FX equities central bank monetary policy"

    def analyze(self, question: str, feedback: str = "", market_moves: str = None) -> Dict[str, Any]:
        period = _extract_period(question)
        macro_query = self._build_macro_query(question)

        # ── Narrative: PRE-filtered vector search (the right month, prose only) ──
        pre_filter: dict = {"content_type": {"$ne": "data_table"}}
        if period:
            pre_filter["report_month"] = {"$eq": period}
        docs = self.context_store.similarity_search(
            macro_query, k=self.context_k, pre_filter=pre_filter
        )

        # ── Distinct source documents behind this retrieval (relevance order) ──
        # Dedupe chunks down to the different PDFs they came from; keep the top 5.
        source_docs: list[str] = []
        for doc in docs:
            src = doc.metadata.get("source")
            if src and src not in source_docs:
                source_docs.append(src)
        source_docs = source_docs[:5]

        # ── Honesty: if a period was asked for but has no context, say so ──
        if period and not docs:
            return {
                "agent": "MarketContextAgent",
                "analysis": (
                    f"No market context is available for {period} in the database. "
                    f"I won't substitute commentary from other periods."
                ),
                "period": period,
                "sources": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        month_label = macro_query.split(" macro")[0] if period else "the relevant period"

        # ── Exact numbers for the period (optional — empty before backfill) ──
        # A/B toggle: set USE_EXACT_TABLES=false to run narrative-only (no daily_table
        # numbers) for comparison against the with-numbers version.
        use_tables = os.getenv("USE_EXACT_TABLES", "true").lower() == "true"
        tables_block, data_warnings = (
            self._get_period_tables(period) if (period and use_tables) else ("", [])
        )

        # ── Recent market moves for the period (computed, deterministic) ──
        # Passed in for override/testing; otherwise computed from market_series.
        if market_moves is None:
            market_moves = self.series.summarize(question).get("moves_block", "")

        system_prompt = (
            "You are a macro market analyst for a hedge fund. "
            "You will be given source documents and, when available, an EXACT MARKET DATA "
            "block of verified closing levels and a RECENT MARKET MOVES block of computed "
            "changes over the period. "
            "Your ONLY job is to summarize what those documents say. "
            "Every single claim you make must be directly supported by the provided material. "
            "Do NOT use any knowledge from your training data. "
            "Do NOT invent events, figures, yields, price moves, or central bank actions. "
            "When citing specific levels or numbers, use the EXACT MARKET DATA figures verbatim — "
            "never estimate a number that is given there. "
            "The RECENT MARKET MOVES figures are pre-computed changes — use them verbatim when "
            "describing how instruments moved over the period; do not recalculate them. "
            "If a value is shown as '?' or a table is marked with a DATA NOTE, treat it as "
            "missing — do not guess it, and briefly flag to the reader that some data was "
            "incomplete for that table. "
            "If a topic is not covered in the documents, do not mention it."
        )

        user_prompt = f"""
Source documents from {month_label}:

{chr(10).join(f'---{chr(10)}{doc.page_content}' for doc in docs)}
{f'''
---
EXACT MARKET DATA (use these figures verbatim for any specific levels):
{tables_block}''' if tables_block else ''}
{f'''
---
{market_moves}''' if market_moves else ''}

---
Based ONLY on the material above, summarize:
1. The key macro events and market-moving developments
2. Which asset classes were affected and how (rates, FX, equities, commodities)
3. Any central bank actions or policy shifts mentioned

Do not add anything not stated above.
{f'''
---
REVISION FEEDBACK FROM USER:
{feedback}
Address this feedback in your revised response. Keep everything that was correct.''' if feedback else ''}
"""

        analysis = self._call_claude(system_prompt, user_prompt)

        return {
            "agent": "MarketContextAgent",
            "analysis": analysis,
            "period": period,
            "has_exact_data": bool(tables_block),
            "has_market_moves": bool(market_moves),
            "data_warnings": data_warnings,   # surfaced to the user if numbers were incomplete
            "sources": source_docs,           # top-5 distinct documents behind the retrieval
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================
# Portfolio Performance Agent
# ============================================================

_MONTH_MAP = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09","sep": "09",  "sept": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}


def _extract_period(question: str) -> str | None:
    """Extract a YYYY-MM period string from a free-text question."""
    q = question.lower()
    for name, num in _MONTH_MAP.items():
        m = re.search(rf'\b{name}\b[^a-z]*\b(20\d{{2}})\b', q)
        if m:
            return f"{m.group(1)}-{num}"
    m = re.search(r'\b(20\d{2})-(\d{2})\b', q)
    if m:
        return m.group(0)
    return None


class PortfolioPerformanceAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Direct MongoDB connection — no vector store needed for structured PnL
        self._mongo_client = MongoClient(os.getenv("MONGO_URI_USER"))

    def _pnl_col(self):
        return self._mongo_client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["pnl_table"]

    def _summary_col(self):
        return self._mongo_client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["pnl_summary"]

    def _get_pnl_summary(self, period: str) -> dict | None:
        doc = self._summary_col().find_one({"report_period": period}, {"_id": 0})
        return doc

    def _extract_period(self, question: str) -> str | None:
        return _extract_period(question)

    def _get_available_periods(self) -> list[str]:
        return sorted(self._pnl_col().distinct("report_period"))

    def _format_as_table(self, rows: list[dict]) -> str:
        """Render row dicts as a markdown table for the prompt."""
        skip = {"report_period", "source_file", "uploaded_by", "uploaded_at", "_id"}
        cols = [k for k in rows[0].keys() if k not in skip]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        lines = [header, sep]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
        return "\n".join(lines)

    def analyze(self, question: str, feedback: str = "") -> Dict[str, Any]:
        period = self._extract_period(question)
        available = self._get_available_periods()

        if not period:
            period = available[-1] if available else None

        if not period:
            return {
                "agent": "PortfolioPerformanceAgent",
                "analysis": "No PnL data available in the database.",
                "period": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        rows = list(self._pnl_col().find({"report_period": period}, {"_id": 0}))

        if not rows:
            return {
                "agent": "PortfolioPerformanceAgent",
                "analysis": (
                    f"No PnL data found for period '{period}'. "
                    f"Available periods: {available}"
                ),
                "period": period,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Fetch pre-computed AUM summary for accurate return figure
        pnl_summary = self._get_pnl_summary(period)
        if pnl_summary and pnl_summary.get("return_pct") is not None:
            ret_pct = pnl_summary["return_pct"]
            start_aum = pnl_summary.get("start_aum", 0)
            end_aum = pnl_summary.get("end_aum", 0)
            total_pnl = pnl_summary.get("total_pnl", 0)
            summary_line = (
                f"PORTFOLIO RETURN FOR {period}: {ret_pct:+.2f}%\n"
                f"Start AUM: ${start_aum:,.0f}  |  End AUM: ${end_aum:,.0f}  |  "
                f"Total P&L: ${total_pnl:,.0f}\n"
                f"This return figure is mathematically exact — use it verbatim."
            )
        else:
            summary_line = ""

        system_prompt = (
            "You are a portfolio performance analyst for a macro hedge fund. "
            "Your job is to produce a thematic analysis of the portfolio — NOT a line-by-line attribution table. "
            "\n\n"
            "GROUP positions by theme: (1) Rates & Duration (include TIPS, swaptions, SOFR futures, Gilts, JGBs together), "
            "(2) Foreign Exchange, (3) Equities, (4) Commodities & Precious Metals, (5) Digital Assets. "
            "Within each theme, lead with total P&L impact, then explain the market dynamics, then describe any trading activity from the Trading Notes. "
            "\n\n"
            "For every major P&L item (>$200k in either direction), articulate the FULL causal chain: "
            "(1) what happened in the market with specific dates, "
            "(2) why it happened — the catalyst, "
            "(3) HOW that move transmitted into the position's P&L — the mechanism through rates, vol, FX, or beta, "
            "(4) any action taken per the Trading Notes and its rationale. "
            "\n\n"
            "Treat linked positions as single trades (e.g. long Gilts + short US 10y = one RV trade). "
            "Distinguish between organic position changes (delta drift, roll) and active trading decisions. "
            "Use exact P&L figures from the data — never round or estimate. "
            "Omit positions with P&L below $200k unless they are strategically relevant."
        )

        user_prompt = f"""
Question: {question}

{summary_line}

P&L Data for period {period}:
{self._format_as_table(rows)}
{f'''
---
REVISION FEEDBACK FROM USER:
{feedback}
Address this feedback in your revised response. Keep everything that was correct.''' if feedback else ''}
"""

        rubric = (
            "1. Positions are grouped by theme (Rates & Duration, FX, Equities, Commodities, Digital Assets) — not listed individually.\n"
            "2. Every P&L figure above $200k includes a full causal chain: catalyst → market move → transmission mechanism → position impact.\n"
            "3. Linked positions (e.g. Gilts + US 10y RV, TIPS curve) are described as single trades, not separately.\n"
            "4. Trading Notes from the data are reflected with their rationale.\n"
            "5. P&L figures are exact — not rounded or estimated.\n"
            "6. Positions below $200k P&L are omitted unless strategically relevant."
        )

        analysis, critique_log = self._call_claude_with_critique(
            system_prompt, user_prompt, rubric, max_retries=1
        )

        return {
            "agent": "PortfolioPerformanceAgent",
            "analysis": analysis,
            "period": period,
            "pnl_summary": pnl_summary,
            "critique_log": critique_log,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================
# Risk Agent
# ============================================================

class RiskAnalystAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mongo_client = MongoClient(os.getenv("MONGO_URI_USER"))

    def _pnl_col(self):
        return self._mongo_client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["pnl_table"]

    def _summary_col(self):
        return self._mongo_client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["pnl_summary"]

    @staticmethod
    def _parse_val(s) -> float | None:
        """Parse a stored value string like '$-2,538,840' or '95000.0' → float."""
        if s is None:
            return None
        s = str(s).strip().replace("$", "").replace(",", "")
        s = re.sub(r"^\((.+)\)$", r"-\1", s)   # (1234) → -1234
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    def _compute_risk_metrics(self, period: str) -> str:
        rows = list(self._pnl_col().find({"report_period": period}, {"_id": 0}))
        if not rows:
            return ""

        summary_doc = self._summary_col().find_one(
            {"report_period": period}, {"_id": 0}
        )
        total_pnl_abs = abs(self._parse_val(
            (summary_doc or {}).get("total_pnl", 0)
        ) or 1)  # avoid div/0

        # ── detect column names (they include the month, e.g. trading_notes_feb_2026)
        sample = rows[0]
        pnl_col = next((k for k in sample if "p_l" in k or "pnl" in k.replace("_","")), None)
        beg_col = next((k for k in sample if "beginning" in k), None)
        end_col = next((k for k in sample if "ending" in k), None)
        type_col = next((k for k in sample if "position_type" in k or k == "type"), None)
        class_col = next((k for k in sample if "asset_class" in k or k == "class"), None)
        pos_col = next((k for k in sample if k == "positions"), None)

        # ── P&L by asset class
        theme_map = {
            "rate": "Rates & Duration",
            "rates": "Rates & Duration",
            "equity": "Equities",
            "fx": "Foreign Exchange",
            "commodity": "Commodities",
            "crypto": "Digital Assets",
            "": "Other",
        }
        theme_pnl: dict[str, float] = {}
        position_pnls: list[tuple[str, str, float]] = []  # (name, theme, pnl)

        for row in rows:
            raw_class = str(row.get(class_col, "") or "").strip().lower()
            theme = theme_map.get(raw_class, "Other")
            pnl_val = self._parse_val(row.get(pnl_col)) if pnl_col else None
            pos_name = str(row.get(pos_col, "Unknown") or "Unknown").strip()

            if pnl_val is not None:
                theme_pnl[theme] = theme_pnl.get(theme, 0.0) + pnl_val
                position_pnls.append((pos_name, theme, pnl_val))

        # ── Rates DV01: net, gross, and curve bucket breakdown
        # Curve bucket classification by position name keywords (heuristic).
        # Front end 0–2y: SOFR futures, NZD front end, FF futures, MUNI ratios
        # Belly 2–10y:    TY (10y Treasury), Gilts (10y), Receiver swaptions (2y1y)
        # Long end 10y+:  20yr TIPS, 20yr swap spreads, JGB 30y
        _BUCKET_KEYWORDS = {
            "front_end": ["sofr", "nzd", "ff futures", "muni", "front end", "green pack"],
            "belly": ["ty", "gilt", "swaption", "jgb 10"],
            "long_end": ["tips", "20yr", "20y", "jgb 30", "swap spread"],
        }

        dv01_beg = 0.0
        dv01_end = 0.0
        dv01_gross = 0.0 # sum of |DV01| — risk magnitude
        dv01_buckets: dict[str, float] = {# ending DV01 by curve bucket
            "front_end": 0.0, "belly": 0.0, "long_end": 0.0, "unclassified": 0.0
        }
        dv01_options = 0.0 # delta-DV01 from options/swaptions

        if beg_col and end_col and type_col:
            for row in rows:
                if str(row.get(type_col, "")).strip().upper() != "DV01":
                    continue
                b = self._parse_val(row.get(beg_col))
                e = self._parse_val(row.get(end_col))
                name_lower = str(row.get(pos_col, "") or "").lower()
                if b is not None:
                    dv01_beg += b
                if e is not None:
                    dv01_end += e
                    dv01_gross += abs(e)

                    # Classify into curve bucket
                    bucket = "unclassified"
                    for bkt, keywords in _BUCKET_KEYWORDS.items():
                        if any(kw in name_lower for kw in keywords):
                            bucket = bkt
                            break
                    dv01_buckets[bucket] += e

                    # Flag options (delta-DV01 is not the same as linear DV01)
                    if "swaption" in name_lower or "option" in name_lower:
                        dv01_options += abs(e if e is not None else 0)

        # ── Notional positions (top 3 longs and shorts by absolute ending notional)
        notional_rows: list[tuple[str, float]] = []
        if end_col and type_col:
            for row in rows:
                if str(row.get(type_col, "")).strip().upper() == "NOTIONAL":
                    e = self._parse_val(row.get(end_col))
                    name = str(row.get(pos_col, "?") or "?").strip()
                    if e is not None and e != 0:
                        notional_rows.append((name, e))
        notional_rows.sort(key=lambda x: x[1], reverse=True)
        top_longs  = [(n, v) for n, v in notional_rows if v > 0][:3]
        top_shorts = [(n, v) for n, v in notional_rows if v < 0][:3]

        # ── Top contributors / detractors
        position_pnls.sort(key=lambda x: x[2], reverse=True)
        contributors = position_pnls[:3]
        detractors = position_pnls[-3:][::-1]

        # ── Format output
        lines = [f"PORTFOLIO RISK METRICS — {period}", ""]

        # Section 1: P&L attribution
        lines.append("── P&L ATTRIBUTION BY THEME ──")
        lines.append(f"{'Theme':<25} {'P&L':>14}  {'% of Total':>10}  Direction")
        lines.append("-" * 60)
        for theme, val in sorted(theme_pnl.items(), key=lambda x: -abs(x[1])):
            pct = (val / total_pnl_abs * 100) if total_pnl_abs else 0
            direction = "▲ Long" if val > 0 else "▼ Short"
            lines.append(f"{theme:<25} ${val:>13,.0f}  {pct:>+9.1f}%  {direction}")
        total_computed = sum(theme_pnl.values())
        lines.append("-" * 60)
        lines.append(f"{'TOTAL (computed)':<25} ${total_computed:>13,.0f}")
        lines.append("")

        # Section 2: Rates DV01
        if dv01_end:
            lines.append("── RATES DV01 EXPOSURE ──")
            lines.append(f"Net DV01  (directional, all same sign = all long):  ${dv01_end:>12,.0f}")
            lines.append(f"Gross DV01 (risk magnitude, sum of |DV01|):         ${dv01_gross:>12,.0f}")
            net_change = dv01_end - dv01_beg
            lines.append(f"Change vs period start:                             ${net_change:>+12,.0f}")
            lines.append("")
            lines.append("Curve bucket breakdown (ending DV01):")
            bucket_labels = {
                "front_end": "Front end 0–2y  (SOFR, NZD, FF)",
                "belly": "Belly 2–10y TY, Gilts, Swaptions)",
                "long_end": "Long end 10y+ (TIPS, Swap Spreads, JGB)",
                "unclassified": "Unclassified",
            }
            for bkt, label in bucket_labels.items():
                val = dv01_buckets[bkt]
                if val:
                    lines.append(f"{label:<42} ${val:>12,.0f}")
            if dv01_options:
                lines.append("")
                lines.append(f"Of which delta-DV01 from options/swaptions: ${dv01_options:>12,.0f}")
                lines.append(f"(delta-DV01 only — vega exposure requires vol surface data not shown here)")
            lines.append("")

        # Section 3: Notional concentration
        if top_longs or top_shorts:
            lines.append("── LARGEST NOTIONAL POSITIONS (ending) ──")
            if top_longs:
                lines.append("  Top longs:")
                for name, val in top_longs:
                    lines.append(f"{name:<35} ${val:>14,.0f}")
            if top_shorts:
                lines.append("Top shorts:")
                for name, val in top_shorts:
                    lines.append(f"{name:<35} ${val:>14,.0f}")
            lines.append("")

        # Section 4: Contributors / detractors
        lines.append("── TOP P&L CONTRIBUTORS ──")
        for name, theme, val in contributors:
            lines.append(f"{name:<35} ${val:>+14,.0f}  ({theme})")
        lines.append("")
        lines.append("── TOP P&L DETRACTORS ──")
        for name, theme, val in detractors:
            lines.append(f"{name:<35} ${val:>+14,.0f}  ({theme})")
        lines.append("")

        # Section 5: Concentration flag
        top3_pnl = sum(abs(p[2]) for p in contributors)
        concentration_pct = top3_pnl / total_pnl_abs * 100 if total_pnl_abs else 0
        lines.append(f"── CONCENTRATION ──")
        lines.append(f"Top 3 positions account for {concentration_pct:.0f}% of gross P&L")

        return "\n".join(lines)

    # ----------------------------------------------------------
    # Stress test helpers
    # ----------------------------------------------------------

    def _get_dominant_themes(self, period: str) -> list[str]:
        """Return the top themes by absolute P&L for scenario selection."""
        rows = list(self._pnl_col().find({"report_period": period}, {"_id": 0}))
        if not rows:
            return []
        sample = rows[0]
        pnl_col = next((k for k in sample if "p_l" in k or "pnl" in k.replace("_", "")), None)
        class_col = next((k for k in sample if "asset_class" in k or k == "class"), None)
        theme_map = {
            "rate": "Rates & Duration", "rates": "Rates & Duration",
            "equity": "Equities", "fx": "Foreign Exchange",
            "commodity": "Commodities", "crypto": "Digital Assets",
        }
        theme_pnl: dict[str, float] = {}
        for row in rows:
            raw = str(row.get(class_col, "") or "").strip().lower()
            theme = theme_map.get(raw, "Other")
            val = self._parse_val(row.get(pnl_col)) if pnl_col else None
            if val is not None:
                theme_pnl[theme] = theme_pnl.get(theme, 0.0) + val
        return [t for t, _ in sorted(theme_pnl.items(), key=lambda x: -abs(x[1]))]

    # Ticker substrings → which scenario move field to use
    _TICKER_MOVE_MAP = {
        "CNH": "usdcnh_pct",
        "HKD": "usd_dxy_pct",
        "CHF": "chfjpy_pct",
        "JPY": "chfjpy_pct",
        "EURGBP": "usd_dxy_pct",
        "PLJ": "platinum_pct",   # platinum futures
        "GDX": "gold_pct",       # gold miners
        "CCJ": "commodities_pct",
        "HGA": "commodities_pct",  # copper
        "IBIT": "crypto_pct",
        "ES": "equities_pct",
        "DEDZ": "equities_pct",
    }
    _CLASS_MOVE_MAP = {
        "rate": "us_10y_bps",
        "rates": "us_10y_bps",
        "equity": "equities_pct",
        "fx": "usd_dxy_pct",
        "commodity": "commodities_pct",
        "crypto": "crypto_pct",
    }

    def _compute_scenario_impacts(
        self, rows: list[dict], scenario: dict
    ) -> dict:
        """
        Compute approximate P&L impact of a scenario on each position.
        Returns a dict with position_impacts list and net_impact float.
        """
        sample = rows[0]
        end_col = next((k for k in sample if "ending" in k), None)
        type_col = next((k for k in sample if "position_type" in k or k == "type"), None)
        class_col = next((k for k in sample if "asset_class" in k or k == "class"), None)
        pos_col = next((k for k in sample if k == "positions"), None)
        tick_col = next((k for k in sample if "ticker" in k), None)

        moves = scenario["market_moves"]
        position_impacts = []
        net_impact = 0.0

        for row in rows:
            pos_name = str(row.get(pos_col, "?") or "?").strip()
            asset_class = str(row.get(class_col, "") or "").strip().lower()
            pos_type = str(row.get(type_col, "") or "").strip().upper()
            ticker = str(row.get(tick_col, "") or "").strip().upper()
            exposure = self._parse_val(row.get(end_col)) if end_col else None

            if exposure is None or exposure == 0:
                continue

            impact = None
            exposure_str = ""
            move_str = ""

            if pos_type == "DV01":
                rate_move = moves.get("us_10y_bps")
                if rate_move is not None:
                    # Long duration: positive DV01 loses when rates rise
                    impact = -exposure * rate_move
                    exposure_str = f"${exposure:>10,.0f} DV01"
                    move_str = f"{rate_move:+d}bps"

            elif pos_type == "NOTIONAL":
                # Try ticker-specific override first
                move_field = next(
                    (v for k, v in self._TICKER_MOVE_MAP.items() if k in ticker),
                    None
                )
                if move_field is None:
                    move_field = self._CLASS_MOVE_MAP.get(asset_class)

                if move_field:
                    pct = moves.get(move_field)
                    if pct is not None:
                        impact = exposure * (pct / 100)
                        exposure_str = f"${exposure:>12,.0f} notl"
                        move_str = f"{pct:+.0f}%"

            if impact is not None:
                position_impacts.append({
                    "position": pos_name,
                    "exposure": exposure_str,
                    "move": move_str,
                    "impact": impact,
                })
                net_impact += impact

        position_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return {
            "scenario": scenario,
            "position_impacts": position_impacts,
            "net_impact": net_impact,
        }

    @staticmethod
    def _format_impact_table(result: dict) -> str:
        """Format a single scenario's computed impacts as a readable table."""
        s = result["scenario"]
        impacts = result["position_impacts"]
        net = result["net_impact"]

        lines = [
            f"SCENARIO: {s['name']}  ({s['period']})",
            f"Context: {s['context']}",
            "",
            f"{'Position':<38} {'Exposure':>18}  {'Move':>8}  {'Est. P&L Impact':>16}",
            "-" * 84,
        ]
        for p in impacts:
            if abs(p["impact"]) < 50_000:
                continue  # skip negligible impacts
            lines.append(
                f"{p['position']:<38} {p['exposure']:>18}  {p['move']:>8}  "
                f"${p['impact']:>+14,.0f}"
            )
        lines += [
            "-" * 84,
            f"{'NET ESTIMATED IMPACT':<38} {'':>18}  {'':>8}  ${net:>+14,.0f}",
            "",
            f"Note: DV01-based estimates assume parallel shift; notional estimates use",
            f"historical scenario peak-to-trough moves. Actual impact depends on timing",
            f"and position delta at time of shock.",
        ]
        return "\n".join(lines)

    def run_stress_test(
        self,
        period: str,
        question: str,
        market_context: str,
        portfolio_performance: str,
        scenario_keys: list[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Run enhanced stress test with curated scenarios and first-derivative
        dollar impact estimates.  Called separately from analyze() — only
        runs when the user opts in via the UI checkbox.

        scenario_keys: list of keys from STRESS_SCENARIOS (e.g. ["2022_rate_shock"]).
                       If None, auto-selects 2 most relevant based on portfolio themes.
        """
        from stress_scenarios import STRESS_SCENARIOS, select_scenarios, format_scenario_for_prompt

        rows = list(self._pnl_col().find({"report_period": period}, {"_id": 0}))
        if not rows:
            return {
                "agent": "StressTest",
                "error": f"No portfolio data for period {period}",
                "period": period,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Select scenarios
        if scenario_keys:
            scenarios = [STRESS_SCENARIOS[k] for k in scenario_keys if k in STRESS_SCENARIOS]
        else:
            dominant = self._get_dominant_themes(period)
            scenarios = select_scenarios(dominant, n=2)

        if not scenarios:
            scenarios = [STRESS_SCENARIOS["2022_rate_shock"]]

        # Compute dollar impacts for each scenario
        impact_results = [self._compute_scenario_impacts(rows, s) for s in scenarios]
        tables_block = "\n\n".join(self._format_impact_table(r) for r in impact_results)
        scenario_block = "\n\n".join(format_scenario_for_prompt(s) for s in scenarios)

        system_prompt = (
            "You are a senior risk analyst for a global macro hedge fund. "
            "You have been given pre-computed first-derivative P&L impact estimates "
            "for specific historical stress scenarios applied to the current portfolio. "
            "Your job is to write a concise stress test interpretation (400–600 words).\n\n"

            "REQUIRED STRUCTURE:\n"
            "1. For each scenario: explain the transmission mechanism for the 2–3 largest impacts. "
            "State which positions drive the loss and through what specific channel "
            "(e.g. parallel rate shift hitting delta-DV01, not vega).\n"
            "2. Correlation regime: explicitly state what correlation is assumed "
            "(e.g. USD strengthens vs CNH in a rate shock) AND what breaks that assumption "
            "(e.g. PBOC intervention, capital controls). Quantify if possible.\n"
            "3. Tail risk: identify the single largest unhedged exposure.\n"
            "4. Risk management: frame recommendations systemically across three dimensions — "
            "(a) directional duration risk, (b) convexity profile (do positions gain from large moves?), "
            "(c) correlation risk (what hedges fail if correlations flip?).\n\n"

            "HARD CONSTRAINTS — violating any of these is an error:\n"
            "- DO NOT describe USDHKD as a hedge to rate shocks. "
            "The HKD peg mechanically caps upside and peg credibility is a separate risk.\n"
            "- DO NOT describe USDCNH as a structural hedge. "
            "CNH behavior is policy-dependent and regime-specific — frame as partial and conditional.\n"
            "- DO NOT attribute swaption losses to vega compression unless vol surface data is provided. "
            "The DV01 figures provided are delta-DV01 only. Default to delta as the driver.\n"
            "- DO NOT infer or speculate about total AUM from position sizes. "
            "Use only figures explicitly provided in the metrics.\n"
            "- DO NOT aggregate DV01 to a single portfolio number. "
            "Reference the curve bucket breakdown (front end / belly / long end) when discussing rate risk.\n"
            "- Risk recommendations must go beyond position cuts. "
            "Propose convex hedges (e.g. payer swaptions, tail puts) and address correlation risk explicitly.\n\n"

            "DO NOT repeat numbers from the impact table — it is shown separately above the narrative. "
            "Reference position names directly. Every claim must be grounded in the provided data."
        )

        user_prompt = f"""
Question: {question}

COMPUTED IMPACT TABLES:
{tables_block}

SCENARIO DETAILS:
{scenario_block}

Market Context:
{market_context}

Portfolio Performance:
{portfolio_performance}

Write the stress test interpretation (400–600 words).
"""

        narrative = self._call_claude(system_prompt, user_prompt)

        return {
            "agent": "StressTest",
            "tables": tables_block,
            "narrative": narrative,
            "scenarios_used": [s["name"] for s in scenarios],
            "period": period,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ----------------------------------------------------------
    # Main analysis
    # ----------------------------------------------------------

    def analyze(self, question: str, market_context: str, portfolio_performance: str, feedback: str = "") -> Dict[str, Any]:
        period = _extract_period(question)

        # Pre-compute structured metrics from pnl_table
        metrics_block = ""
        if period:
            try:
                metrics_block = self._compute_risk_metrics(period)
            except Exception as e:
                print(f"  RiskAnalystAgent: metrics computation failed — {e}")

        system_prompt = (
            "You are a senior portfolio risk analyst for a global macro hedge fund. "
            "You will be given structured portfolio metrics, market context, and performance analysis. "
            "Produce a rigorous, forward-looking risk report organized into EXACTLY FOUR sections:\n\n"

            "SECTION 1 — MACRO DRIVERS\n"
            "Identify the 2–3 most influential macro forces that drove portfolio performance this period. "
            "For each driver: name the specific force (e.g. 'US real rate decline', 'CNH devaluation pressure'), "
            "explain which positions it affected and through what transmission mechanism, "
            "and identify cross-asset correlations — which positions were driven by the same underlying force "
            "and which moved independently. "
            "Use the P&L attribution table to quantify how much each driver contributed.\n\n"

            "SECTION 2 — DIVERSIFICATION OPPORTUNITIES\n"
            "Given the current position concentrations shown in the metrics, screen for 2–3 trade ideas "
            "that are structurally uncorrelated or negatively correlated with the dominant exposures. "
            "For each idea: name the instrument or theme, explain WHY it is uncorrelated under the current regime, "
            "and describe what macro scenario it hedges against. "
            "Do not recommend adding more of what already worked — focus on genuine diversifiers.\n\n"

            "SECTION 3 — POSITION ADJUSTMENTS\n"
            "Propose 2–3 specific adjustments to improve the portfolio's risk-adjusted return profile. "
            "Reference the concentration data and the top detractors directly. "
            "For each adjustment: state what to do (add, reduce, restructure), name the position, "
            "explain the risk-return rationale, and identify what would trigger the adjustment. "
            "Be concrete — avoid generic 'reduce risk' language.\n\n"

            "SECTION 4 — STRESS TESTS\n"
            "Run three stress tests: two historical analogs and one forward-looking hypothetical. "
            "For each test: name the scenario, describe the key market moves (rates, FX, equities, vol), "
            "identify which specific positions are most exposed using the DV01 and notional data provided, "
            "estimate the approximate P&L impact direction and magnitude for each major theme, "
            "and note any natural hedges that would offset the shock. "
            "Historical analogs should be chosen for their relevance to the CURRENT portfolio composition — "
            "not just the most famous crises. The hypothetical should reflect a plausible tail risk "
            "in the current macro environment.\n\n"

            "STANDARDS: Use exact figures from the metrics table. Never invent data. "
            "Every claim about a position must reference the actual position name from the data. "
            "Avoid generic risk language — every sentence should be specific to this portfolio."
        )

        user_prompt = f"""
Question: {question}

{f"PORTFOLIO RISK METRICS:{chr(10)}{metrics_block}{chr(10)}" if metrics_block else ""}
Market Context:
{market_context}

Portfolio Performance:
{portfolio_performance}
{f'''
---
REVISION FEEDBACK FROM USER:
{feedback}
Address this feedback in your revised response. Keep everything that was correct.''' if feedback else ''}
"""

        analysis = self._call_claude(system_prompt, user_prompt)

        return {
            "agent": "RiskAnalystAgent",
            "analysis": analysis,
            "metrics": metrics_block,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================
# Newsletter Writer
# ============================================================

class NewsletterWriterAgent(BaseAgent):
    def write(
        self,
        question: str,
        market_context: str,
        portfolio_performance: str,
        risk_analysis: str,
        pnl_summary: dict | None = None,
        feedback: str = "",
    ) -> Dict[str, Any]:
        # Build a return fact line to anchor the opening paragraph
        if pnl_summary and pnl_summary.get("return_pct") is not None:
            ret_pct = pnl_summary["return_pct"]
            total_pnl = pnl_summary.get("total_pnl", 0)
            return_fact = (
                f"EXACT MONTHLY RETURN: {ret_pct:+.2f}%  (Total P&L: ${total_pnl:,.0f})\n"
                f"You MUST use this figure in the opening paragraph. Do not calculate or estimate the return yourself."
            )
        else:
            return_fact = ""

        system_prompt = (
            "You are writing a monthly investor letter for a global macro hedge fund. "
            "Target length: 800–1,200 words. Tone: direct, candid, analytical — modest on the surface "
            "but firm in conviction; not defensive or euphemistic. "
            "\n\n"
            "This letter is a SUMMARY and narrative. The detailed numbers already live in the Performance "
            "and Risk sections, so do NOT reproduce them here — the reader has them elsewhere.\n"
            "\n\n"
            "STRUCTURE:\n"
            "1. Opening paragraph: state the headline monthly return ONCE, then a 2-3 sentence macro summary of what drove the month.\n"
            "2. Body: organized by THEME not by position. Suggested sections: Precious Metals & Miners, "
            "Rates & Duration, Foreign Exchange, Equities, Digital Assets. "
            "Within each section: explain the market dynamic and how it affected the book, then the thesis. "
            "Refer to P&L magnitudes QUALITATIVELY ('the largest contributor', 'a modest drag', 'roughly offset') "
            "rather than quoting exact dollar figures.\n"
            "3. Outlook: connect current positioning to specific forward catalysts. "
            "Name what needs to happen for the portfolio to benefit AND what would make the thesis wrong.\n"
            "4. Closing: one short paragraph.\n"
            "\n\n"
            "NUMBERS POLICY:\n"
            "- Use the headline monthly return figure in the opening — that one figure is expected.\n"
            "- Beyond that, AVOID specific dollar P&L figures and line-item attribution. Speak in relative terms.\n"
            "- Do not duplicate the exact numbers that already appear in the Performance and Risk sections.\n"
            "\n\n"
            "ANALYTICAL STANDARDS:\n"
            "- Explain transmission mechanisms, not just correlations. "
            "Don't say 'oil rose and gold fell' — explain the full chain: why oil moved, how that repriced inflation/rates, how that mechanism hit the position.\n"
            "- Surface cross-asset linkages: which positions were driven by the same underlying force? Which provided natural offsets?\n"
            "- Organize around REGIME PHASES within the month (e.g. 'Early Feb: gold volatility phase, Feb 2–7'), not day-by-day recitation.\n"
            "- Do NOT list every line item. This is a narrative, not an attribution table.\n"
            "- Never assert macro facts (data releases, yield levels) not present in the provided context."
        )

        user_prompt = f"""
Question: {question}

{return_fact}

Market Context:
{market_context}

Portfolio Performance:
{portfolio_performance}

Risk Analysis:
{risk_analysis}

Write a professional monthly newsletter.
{f'''
---
REVISION FEEDBACK FROM USER:
{feedback}
Address this feedback in your revised response. Keep everything that was correct.''' if feedback else ''}
"""

        rubric = (
            "1. Opens with the headline monthly return figure and a 2-3 sentence macro summary.\n"
            "2. Body is organized by theme/asset class — NOT a position-by-position list.\n"
            "3. P&L is described QUALITATIVELY (relative magnitudes) — no line-item dollar figures beyond the headline return.\n"
            "4. Transmission mechanisms are explained (full causal chain), not just correlations stated.\n"
            "5. Cross-asset linkages are surfaced (which positions were driven by the same force).\n"
            "6. Month is organized around regime phases, not a chronological day-by-day walkthrough.\n"
            "7. Outlook names specific catalysts and acknowledges what would make the thesis wrong.\n"
            "8. No macro facts asserted without support from the provided context.\n"
            "9. Length is approximately 800–1,200 words."
        )

        newsletter, critique_log = self._call_claude_with_critique(
            system_prompt, user_prompt, rubric, max_retries=1
        )

        return {
            "agent": "NewsletterWriterAgent",
            "newsletter": newsletter,
            "critique_log": critique_log,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================
# Orchestrator
# ============================================================

class OrchestratorAgent:
    def __init__(self, embedding, anthropic_client: Anthropic, context_k: int = 30, model: str = None):
        # Resolve model once here — passed to all agents so they don't each call the API
        self._model = model or os.getenv("CLAUDE_MODEL") or get_latest_opus_model(anthropic_client)
        self._anthropic = anthropic_client
        print(f"  [Orchestrator] Using model: {self._model}")

        self.market = MarketContextAgent(embedding, anthropic_client, context_k, model=self._model)
        self.performance = PortfolioPerformanceAgent(embedding, anthropic_client, context_k, model=self._model)
        self.risk = RiskAnalystAgent(embedding, anthropic_client, context_k, model=self._model)
        self.writer = NewsletterWriterAgent(embedding, anthropic_client, context_k, model=self._model)

    # ----------------------------------------------------------
    # Planner: lightweight intent classification
    # ----------------------------------------------------------

    async def _plan_execution(self, query: str) -> dict:
        """
        One cheap Claude call that reads the query and returns a JSON plan:
        which agents to run, in what order, and what response format to use.

        Falls back to the full pipeline on any failure so the user always
        gets a result.

        Returns:
        {
          "intent":        "newsletter|market_analysis|performance|risk_analysis|
                            stress_test|full_brief|qa",
          "description":   "one-sentence summary of what was requested",
          "period":        "2026-02 | null",
          "response_type": "newsletter|analysis|chat",
          "agents":        ["performance", "market", ...],   # ordered
          "reasoning":     "brief explanation"
        }

        Agent dependency rules (enforced at execution time, not here):
          risk        requires performance + market
          newsletter  requires performance + market + risk
          stress_test requires performance + market
        """
        import json

        # Fetch available periods for the planner's context
        available_periods: list[str] = []
        try:
            from pymongo import MongoClient
            _c = MongoClient(os.getenv("MONGO_URI_USER"))
            _col = _c[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["pnl_table"]
            available_periods = sorted(_col.distinct("report_period"))
        except Exception:
            pass
        periods_str = ", ".join(available_periods) if available_periods else "unknown"

        system_prompt = (
            "You are a routing agent for a portfolio analysis system. "
            "Given a user query, determine exactly what they want and which agents to run.\n\n"
            "Available agents:\n"
            "performance  — portfolio P&L, returns, attribution for a specific period\n"
            "market — macro market context (rates, FX, equities, central banks)\n"
            "risk — risk analysis with DV01, notional, concentration (needs performance + market)\n"
            "newsletter — full investor letter (needs performance + market + risk)\n"
            "stress_test — scenario stress test (needs performance + market)\n\n"
            f"Available data periods: {periods_str}\n\n"
            "Intent → agent mapping (use EXACTLY these agent lists):\n"
            "  newsletter       → [performance, market, risk, newsletter]\n"
            "  market_analysis  → [market]\n"
            "  performance      → [performance]\n"
            "  risk_analysis    → [performance, market, risk]\n"
            "  stress_test      → [performance, market, stress_test]\n"
            "  full_brief       → [performance, market, risk]\n"
            "  qa               → [market]\n\n"
            "response_type rules:\n"
            "  newsletter intent              → response_type: newsletter\n"
            "  market/performance/risk/brief  → response_type: analysis\n"
            "  qa / general question          → response_type: chat\n\n"
            "If the user asks for 'analysis', 'brief', 'summary' without mentioning a newsletter → full_brief.\n"
            "If the user mentions 'stress test' or 'scenario' → stress_test.\n"
            "If unsure → newsletter (safest default).\n\n"
            "Return ONLY valid JSON — no markdown fences, no extra text:\n"
            "{\n"
            '  "intent": "...",\n'
            '  "description": "one sentence",\n'
            '  "period": "YYYY-MM or null",\n'
            '  "response_type": "newsletter|analysis|chat",\n'
            '  "agents": [...],\n'
            '  "reasoning": "brief explanation"\n'
            "}"
        )

        try:
            response = self._anthropic.messages.create(
                model=os.getenv("CLAUDE_MODEL") or get_latest_opus_model(self._anthropic),
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": f"Query: {query}"}],
            )
            text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            ).strip()

            # Extract JSON even if there's minor surrounding text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                plan = json.loads(match.group())
                if "agents" in plan and "response_type" in plan:
                    print(f"  Planner → intent={plan.get('intent')} agents={plan.get('agents')}")
                    return plan
        except Exception as e:
            print(f"  Planner failed ({e}) — using full pipeline fallback")

        # Fallback: full pipeline
        return {
            "intent": "newsletter",
            "description": "Full pipeline (planner fallback)",
            "period": _extract_period(query),
            "response_type": "newsletter",
            "agents": ["performance", "market", "risk", "newsletter"],
            "reasoning": "fallback",
        }

    # ----------------------------------------------------------
    # Flexible entry point
    # ----------------------------------------------------------

    async def run(
        self,
        query: str,
        stress_test_options: dict | None = None,
    ) -> Dict[str, Any]:
        """
        Flexible, intent-driven entry point.

        1. Calls _plan_execution() to determine which agents are needed.
        2. Runs only those agents, in parallel where dependencies allow.
        3. Returns result dict with `response_type` and `plan` metadata.

        stress_test_options (optional, same schema as run_parallel):
            {"period": "2026-02", "scenario_keys": ["2022_rate_shock"]}
        If stress_test is in the plan OR stress_test_options is provided,
        the stress test runs automatically.
        """
        plan = await self._plan_execution(query)
        agents_needed = set(plan.get("agents", []))
        response_type = plan.get("response_type", "newsletter")

        result: Dict[str, Any] = {
            "question": query,
            "plan": plan,
            "response_type": response_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # ── Group 1: independent agents (run in parallel) ──────────
        group1: dict[str, Any] = {}
        if "market" in agents_needed:
            group1["market"] = asyncio.to_thread(self.market.analyze, query)
        if "performance" in agents_needed:
            group1["performance"] = asyncio.to_thread(self.performance.analyze, query)

        if group1:
            gathered = await asyncio.gather(*group1.values())
            for key, res in zip(group1.keys(), gathered):
                result[key] = res

        # ── Group 2: risk (requires performance + market) ──────────
        if "risk" in agents_needed:
            risk_result = await asyncio.to_thread(
                self.risk.analyze,
                query,
                result.get("market", {}).get("analysis", ""),
                result.get("performance", {}).get("analysis", ""),
            )
            result["risk"] = risk_result

        # ── Group 3: newsletter (requires performance + market + risk) ─
        if "newsletter" in agents_needed:
            newsletter_result = await asyncio.to_thread(
                self.writer.write,
                query,
                result.get("market", {}).get("analysis", ""),
                result.get("performance", {}).get("analysis", ""),
                result.get("risk", {}).get("analysis", ""),
                result.get("performance", {}).get("pnl_summary"),
            )
            result["newsletter"] = newsletter_result

        # ── Stress test (explicit options OR planner routed here) ───
        run_stress = stress_test_options or ("stress_test" in agents_needed)
        if run_stress:
            opts = stress_test_options or {}
            period = opts.get("period") or plan.get("period") or _extract_period(query)
            scenario_keys = opts.get("scenario_keys")
            if period:
                # Ensure we have performance + market data for the stress test
                if "performance" not in result:
                    result["performance"] = await asyncio.to_thread(
                        self.performance.analyze, query
                    )
                if "market" not in result:
                    result["market"] = await asyncio.to_thread(
                        self.market.analyze, query
                    )
                result["stress_test"] = await asyncio.to_thread(
                    self.risk.run_stress_test,
                    period,
                    query,
                    result["market"]["analysis"],
                    result["performance"]["analysis"],
                    scenario_keys,
                )

        return result

    # ----------------------------------------------------------
    # Backward-compat alias
    # ----------------------------------------------------------

    async def run_parallel(
        self,
        question: str,
        stress_test_options: dict | None = None,
    ) -> Dict[str, Any]:
        """
        Legacy alias — forces the full newsletter pipeline regardless of intent.
        Prefer run() for intent-aware routing.
        """
        # Override planner by injecting newsletter into the query hint
        _q = question if "newsletter" in question.lower() else f"Write a newsletter. {question}"
        return await self.run(_q, stress_test_options=stress_test_options)

    async def revise_section(
        self,
        section: str,
        feedback: str,
        current_result: dict,
    ) -> dict:
        """
        Re-run a single section with user feedback.

        section options: "newsletter", "risk", "performance", "market"

        Cascades to newsletter ONLY if the current result already has one —
        i.e. if the original query produced a newsletter. For analysis-only
        results, revisions update the section in-place without spawning a
        newsletter the user didn't ask for.
        """
        import copy
        result = copy.deepcopy(current_result)
        question = result["question"]
        has_newsletter = "newsletter" in result  # only cascade if already present

        if section == "newsletter":
            writer_result = await asyncio.to_thread(
                self.writer.write,
                question,
                result.get("market", {}).get("analysis", ""),
                result.get("performance", {}).get("analysis", ""),
                result.get("risk", {}).get("analysis", ""),
                result.get("performance", {}).get("pnl_summary"),
                feedback,
            )
            result["newsletter"] = writer_result

        elif section == "risk":
            risk_result = await asyncio.to_thread(
                self.risk.analyze,
                question,
                result.get("market", {}).get("analysis", ""),
                result.get("performance", {}).get("analysis", ""),
                feedback,
            )
            result["risk"] = risk_result
            if has_newsletter:
                writer_result = await asyncio.to_thread(
                    self.writer.write,
                    question,
                    result.get("market", {}).get("analysis", ""),
                    result.get("performance", {}).get("analysis", ""),
                    risk_result["analysis"],
                    result.get("performance", {}).get("pnl_summary"),
                )
                result["newsletter"] = writer_result

        elif section == "performance":
            perf_result = await asyncio.to_thread(
                self.performance.analyze,
                question,
                feedback,
            )
            result["performance"] = perf_result
            if has_newsletter:
                writer_result = await asyncio.to_thread(
                    self.writer.write,
                    question,
                    result.get("market", {}).get("analysis", ""),
                    perf_result["analysis"],
                    result.get("risk", {}).get("analysis", ""),
                    perf_result.get("pnl_summary"),
                )
                result["newsletter"] = writer_result

        elif section == "market":
            market_result = await asyncio.to_thread(
                self.market.analyze,
                question,
                feedback,
            )
            result["market"] = market_result
            if has_newsletter:
                writer_result = await asyncio.to_thread(
                    self.writer.write,
                    question,
                    market_result["analysis"],
                    result.get("performance", {}).get("analysis", ""),
                    result.get("risk", {}).get("analysis", ""),
                    result.get("performance", {}).get("pnl_summary"),
                )
                result["newsletter"] = writer_result

        # track revision in result metadata
        if "revisions" not in result:
            result["revisions"] = []
        result["revisions"].append({
            "section": section,
            "feedback": feedback,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return result


# ============================================================
# Factory
# ============================================================

def build_agent_system() -> OrchestratorAgent:
    load_dotenv()

    provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()

    if provider == "gemini":
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
    elif provider == "openai":
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    anthropic_client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    # Resolve model once at startup — all agents share this single resolved value
    model = os.getenv("CLAUDE_MODEL") or get_latest_opus_model(anthropic_client)

    return OrchestratorAgent(embedding, anthropic_client, model=model)

