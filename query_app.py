import re
import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from multiagent import build_agent_system, _extract_period
from helper import get_latest_opus_model
from stress_scenarios import STRESS_SCENARIOS
from eval_collector import EvalCollector
import wayfound_monitor as wayfound


def safe_md(text: str) -> str:
    """Escape bare dollar signs so Streamlit doesn't render them as LaTeX."""
    return re.sub(r'\$(?=[\d\s(])', r'\\$', text)


def revision_controls(section_key: str, section_label: str, result_idx: int = -1):
    fb_key  = f"fb_{section_key}"
    btn_key = f"btn_revise_{section_key}"

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Revise this section</div>',
        unsafe_allow_html=True,
    )
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        feedback = st.text_input(
            label="feedback",
            label_visibility="collapsed",
            placeholder=f"Describe what to change in the {section_label}...",
            key=fb_key,
        )
    with col_btn:
        pressed = st.button("Revise", key=btn_key, use_container_width=True)

    if pressed and feedback.strip():
        st.session_state["pending_revision"] = {
            "section": section_key,
            "feedback": feedback.strip(),
        }
        st.rerun()


load_dotenv()

st.set_page_config(
    page_title="Portfolio AI",
    page_icon="📊",
    layout="wide"
)

# -------------------------
# Top Navigation Bar
# -------------------------

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Jost:wght@300;400;500&display=swap" rel="stylesheet">

    <style>
        header[data-testid="stHeader"] { display: none !important; }

        .navbar {
            display: flex;
            align-items: center;
            background-color: #f5f0e8;
            padding: 14px 32px;
            margin: -60px -4rem 32px -4rem;
            border-bottom: 1px solid #ddd5c4;
            gap: 32px;
        }
        .navbar-brand {
            font-family: 'Cormorant Garamond', serif;
            font-weight: 600;
            font-size: 17px;
            color: #2c2c2c;
            margin-right: auto;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .navbar a {
            font-family: 'Cormorant Garamond', serif;
            color: #7a6e60;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            transition: color 0.2s;
        }
        .navbar a:hover { color: #2c2c2c; }
    </style>

    <div class="navbar">
        <span class="navbar-brand">Navigation</span>
        <a href="https://honte-search-app.streamlit.app/" target="_blank">Search</a>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# Custom Styling
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Jost:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Jost', sans-serif;
        background-color: #ffffff;
        color: #2c2c2c;
    }

    .main { background-color: #ffffff; }

    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
    }

    p, .stMarkdown {
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        color: #4a4a4a !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }

    /* Login inputs */
    .stTextInput label {
        font-family: 'Jost', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #7a6e60 !important;
    }

    .stTextInput input {
        background-color: #faf8f5 !important;
        color: #2c2c2c !important;
        border: 1px solid #ddd5c4 !important;
        border-radius: 2px !important;
        font-family: 'Jost', sans-serif !important;
        font-size: 15px !important;
        font-weight: 300 !important;
        box-shadow: none !important;
    }

    .stTextInput input:focus {
        border-color: #b8a99a !important;
        box-shadow: 0 0 0 1px #b8a99a !important;
    }

    /* Label above text area */
    .stTextArea label {
        font-family: 'Jost', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #7a6e60 !important;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: #faf8f5 !important;
        color: #2c2c2c !important;
        border: 1px solid #ddd5c4 !important;
        border-radius: 2px !important;
        font-family: 'Jost', sans-serif !important;
        font-size: 15px !important;
        font-weight: 300 !important;
        box-shadow: none !important;
    }

    .stTextArea textarea:focus {
        border-color: #b8a99a !important;
        box-shadow: 0 0 0 1px #b8a99a !important;
    }

    /* Button */
    .stButton > button {
        background-color: #2c2c2c !important;
        color: #f5f0e8 !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 400 !important;
        font-size: 13px !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 1px !important;
        padding: 0.55rem 2.2rem !important;
        transition: background-color 0.2s !important;
    }

    .stButton > button p {
        color: #f5f0e8 !important;
    }

    .stButton > button:disabled {
        background-color: #2c2c2c !important;
        color: #f5f0e8 !important;
        opacity: 0.6 !important;
    }

    .stButton > button:hover { background-color: #4a4a4a !important; }

    /* Divider */
    hr {
        border: none !important;
        border-top: 1px solid #e8e2d9 !important;
        margin: 2rem 0 !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-family: 'Jost', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #7a6e60 !important;
        background-color: #faf8f5 !important;
        border: 1px solid #e8e2d9 !important;
        border-radius: 1px !important;
    }

    .streamlit-expanderContent {
        background-color: #faf8f5 !important;
        border: 1px solid #e8e2d9 !important;
        border-top: none !important;
        padding: 1.5rem !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        font-size: 15px !important;
        line-height: 1.8 !important;
        color: #2c2c2c !important;
    }

    /* Newsletter output block */
    .newsletter-block {
        background-color: #faf8f5;
        border-left: 2px solid #c9a87a;
        padding: 2rem 2.5rem;
        margin-top: 1rem;
        font-size: 16px;
        line-height: 1.9;
        color: #2c2c2c;
        font-family: 'Jost', sans-serif;
        font-weight: 300;
    }

    /* Subheader label */
    .section-label {
        font-family: 'Jost', sans-serif;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #9a8f82;
        margin-bottom: 0.5rem;
    }

    /* Warning / Alert */
    .stAlert {
        background-color: #faf8f5 !important;
        border: 1px solid #ddd5c4 !important;
        color: #7a6e60 !important;
        border-radius: 1px !important;
    }

    /* Spinner */
    .stSpinner > div { border-top-color: #c9a87a !important; }

    /* Session history item */
    .history-label {
        font-family: 'Jost', sans-serif;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #9a8f82;
        margin-top: 1.5rem;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Login
# ============================================================

from auth_helper import verify_login, is_authenticated

# ── Auth gate ──────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None

if not is_authenticated(st.session_state):
    st.title("Portfolio Newsletter Generator")
    st.divider()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = verify_login(username.strip(), password)
        if user:
            st.session_state.authenticated = True
            st.session_state.username = user["username"]
            st.session_state.role = user["role"]
            st.rerun()
        else:
            st.error("Incorrect username or password.")
    st.stop()

# ============================================================
# Orchestrator — shared across all users, created once
# Uses read-only MongoDB credentials via MONGO_URI in .env
# ============================================================

@st.cache_resource
def get_orchestrator():
    return build_agent_system()

@st.cache_resource
def get_validator():
    """Validation/revision agent — wraps the shared orchestrator. Created once."""
    from validation_agent import ValidationAgent
    orch = get_orchestrator()
    return ValidationAgent(orch, orch._anthropic, orch._model)

@st.cache_data(ttl=300)
def get_available_periods() -> list[str]:
    """Fetch PnL periods from MongoDB — cached for 5 min."""
    from pymongo import MongoClient
    client = MongoClient(os.getenv("MONGO_URI_USER"))
    col = client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["pnl_table"]
    return sorted(col.distinct("report_period"))

orchestrator = get_orchestrator()
validator = get_validator()

# ============================================================
# Per-user session history
# Lost on tab close — nothing written to MongoDB
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# Header
# ============================================================

col1, col2 = st.columns([6, 1])
with col1:
    st.title("Portfolio AI")
    st.markdown("Ask anything — newsletter, risk analysis, market brief, stress test.")
with col2:
    st.markdown("<div style='padding-top: 1.8rem;'></div>", unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.history = []
        st.rerun()

# ============================================================
# Input
# ============================================================

query = st.text_area(
    "Enter Request",
    height=120,
    placeholder=(
        "Examples:\n"
        "• Write a February 2026 investor newsletter\n"
        "• Give me a risk analysis for Feb 2026\n"
        "• What happened in markets in February?\n"
        "• Stress test the portfolio against a rate shock"
    ),
)

# ── Stress test controls ──────────────────────────────────────
run_stress = st.checkbox("Include stress test")

stress_options = None
if run_stress:
    available_periods = get_available_periods()
    auto_period = _extract_period(query) if query.strip() else None

    st.markdown(
        '<div class="section-label" style="margin-top:0.4rem;">Stress test settings</div>',
        unsafe_allow_html=True,
    )
    col_period, col_mode = st.columns([2, 3])

    with col_period:
        period_options = available_periods if available_periods else ["(no data)"]
        default_idx = (
            period_options.index(auto_period)
            if auto_period and auto_period in period_options
            else len(period_options) - 1
        )
        selected_period = st.selectbox(
            "Period",
            options=period_options,
            index=default_idx,
            key="stress_period",
        )

    with col_mode:
        scenario_mode = st.radio(
            "Scenario selection",
            options=["AI auto-select", "Choose manually"],
            horizontal=True,
            key="scenario_mode",
        )

    selected_keys = None
    if scenario_mode == "Choose manually":
        scenario_labels = {v["name"]: k for k, v in STRESS_SCENARIOS.items()}
        chosen = st.multiselect(
            "Select scenarios (choose 1–3)",
            options=list(scenario_labels.keys()),
            default=list(scenario_labels.keys())[:2],
            key="stress_scenarios",
        )
        selected_keys = [scenario_labels[c] for c in chosen] if chosen else None

    if selected_period and selected_period != "(no data)":
        stress_options = {
            "period": selected_period,
            "scenario_keys": selected_keys,
        }

generate = st.button("Generate")

# ============================================================
# Execution — read only, no writes to MongoDB
# ============================================================

if generate and query.strip():
    run_started = wayfound.utc_now_iso()
    with st.spinner("Planning and running agents..."):
        result = asyncio.run(
            orchestrator.run(query, stress_test_options=stress_options)
        )

    # Save to session history (RAM only — private to user)
    hist_entry = {
        "query": query,
        "result": result,
    }

    # Wayfound supervision — one session per query, revisions append to it.
    # No-op unless WAYFOUND_API_KEY is configured; never blocks the UI.
    recorder = wayfound.start_recorder(result, username=st.session_state.username)
    if recorder:
        recorder.record(wayfound.messages_for_run(query, result, run_started))
        hist_entry["wayfound"] = recorder

    st.session_state.history.append(hist_entry)

    # clear any prepared exports — they belong to the previous result
    for k in ("export_pdf", "export_docx", "export_pdf_name", "export_docx_name"):
        st.session_state.pop(k, None)

    # auto-save to eval_examples collection for future rating / fine-tuning
    try:
        category = result.get("plan", {}).get("intent") or result.get("response_type", "unknown")
        EvalCollector().save_example(result, category=category)
    except Exception as _e:
        pass  # never block the UI for this

elif generate:
    st.warning("Please enter a request.")

# ============================================================
# Revision execution 
# ============================================================
if st.session_state.get("pending_revision") and st.session_state.history:
    rev = st.session_state.pop("pending_revision")
    current = st.session_state.history[-1]["result"]
    label_map = {
        "newsletter": "Newsletter",
        "risk": "Risk Analysis",
        "performance": "Portfolio Performance",
        "market": "Market Context",
    }
    label = label_map.get(rev["section"], rev["section"].title())
    rev_started = wayfound.utc_now_iso()
    with st.spinner(f"Revising {label}..."):
        updated = asyncio.run(
            orchestrator.revise_section(rev["section"], rev["feedback"], current)
        )
    st.session_state.history[-1]["result"] = updated

    # Wayfound: append the feedback + only the re-run sections to this query's session
    recorder = st.session_state.history[-1].get("wayfound")
    if recorder:
        recorder.record(wayfound.messages_for_revision(
            rev["feedback"], updated, rev_started,
            section=rev["section"], via="section_revise",
        ))
    # Prepared exports are now stale — drop them
    for k in ("export_pdf", "export_docx", "export_pdf_name", "export_docx_name"):
        st.session_state.pop(k, None)
    st.rerun()

# ============================================================
# Validation agent — execute a confirmed revision plan.
# Snapshots the current result onto the version stack first,
# then runs the plan over the dirty set.
# ============================================================
if st.session_state.get("pending_validation_execute") and st.session_state.history:
    plan = st.session_state.pop("pending_validation_execute")
    hist_entry = st.session_state.history[-1]
    current = hist_entry["result"]

    # Snapshot the pre-revision result onto this entry's version stack
    from validation_agent import ValidationAgent
    hist_entry.setdefault("versions", []).append({
        "label":  ValidationAgent.version_label(plan),
        "result": current,
    })

    exec_started = wayfound.utc_now_iso()
    with st.spinner("Applying revision..."):
        updated = asyncio.run(validator.execute_revision(plan, current))
    hist_entry["result"] = updated

    # Wayfound: append the re-run sections + the agent's own explanation
    recorder = hist_entry.get("wayfound")
    if recorder:
        recorder.record(wayfound.messages_for_validation_execution(updated, exec_started))

    st.session_state.pop("validation_complaint_text", None)  # clear the box
    st.rerun()

def render_plan_badge(result: dict):
    plan = result.get("plan")
    if not plan:
        return
    intent = plan.get("intent", "").replace("_", " ").title()
    agents_run = plan.get("agents", [])
    desc = plan.get("description", "")
    reasoning = plan.get("reasoning", "")
    badge_parts = [f"**{intent}**"]
    if agents_run:
        badge_parts.append(f"agents: {', '.join(agents_run)}")
    if reasoning and reasoning != "fallback":
        badge_parts.append(f"_{reasoning}_")
    st.caption("  ·  ".join(badge_parts))
    if desc:
        st.caption(f"↳ {desc}")


def render_market_section(result: dict, expanded: bool = True, show_controls: bool = True):
    if "market" not in result:
        return
    with st.expander("Market Context Analysis", expanded=expanded):
        st.markdown(safe_md(result["market"]["analysis"]))
        if show_controls:
            revision_controls("market", "Market Context")


def render_performance_section(result: dict, expanded: bool = True, show_controls: bool = True):
    if "performance" not in result:
        return
    with st.expander("Portfolio Performance Analysis", expanded=expanded):
        st.markdown(safe_md(result["performance"]["analysis"]))
        if show_controls:
            revision_controls("performance", "Portfolio Performance")


def render_risk_section(result: dict, expanded: bool = True, show_controls: bool = True):
    if "risk" not in result:
        return
    with st.expander("Risk Analysis", expanded=expanded):
        risk = result["risk"]
        metrics = risk.get("metrics", "")
        if metrics:
            st.markdown(
                '<div class="section-label">Portfolio Metrics</div>',
                unsafe_allow_html=True,
            )
            st.code(metrics, language=None)
            st.divider()
        st.markdown(
            '<div class="section-label">Analysis</div>',
            unsafe_allow_html=True,
        )
        st.markdown(safe_md(risk["analysis"]))
        if show_controls:
            revision_controls("risk", "Risk Analysis")


def render_stress_test_section(result: dict):
    st_result = result.get("stress_test")
    if not st_result:
        return
    if st_result.get("error"):
        with st.expander("Stress Test — Error"):
            st.warning(st_result["error"])
    else:
        label = "Stress Test — " + ", ".join(st_result.get("scenarios_used", []))
        with st.expander(label, expanded=True):
            st.markdown(
                '<div class="section-label">First-Derivative Impact Estimates</div>',
                unsafe_allow_html=True,
            )
            st.code(st_result["tables"], language=None)
            st.divider()
            st.markdown(
                '<div class="section-label">Interpretation</div>',
                unsafe_allow_html=True,
            )
            st.markdown(safe_md(st_result["narrative"]))


def render_sources(result: dict):
    """
    One overall Sources footer for the whole response — top 5 distinct documents
    across every agent that retrieved from the corpus (not per-section). Shown for
    every request type. Silent if nothing was retrieved (e.g. structured-only paths).
    """
    seen: list[str] = []
    for section in result.values():
        if isinstance(section, dict):
            for src in section.get("sources", []) or []:
                if src and src not in seen:
                    seen.append(src)
    seen = seen[:5]
    if not seen:
        return
    lines = "\n".join(f"{i}. {s}" for i, s in enumerate(seen, 1))
    st.markdown("---")
    st.markdown(f"**Sources (top {len(seen)} documents referenced):**\n\n{lines}")


def render_result(result: dict, show_revision_controls: bool = True):
    """
    Render a result dict based on its response_type (newsletter, analysis, chat)
    """
    response_type = result.get("response_type", "newsletter")
    revisions = result.get("revisions", [])

    render_plan_badge(result)

    if revisions:
        rev_summary = ", ".join(
            f"{r['section']} ×{sum(1 for x in revisions if x['section'] == r['section'])}"
            for r in {r['section']: r for r in revisions}.values()
        )
        st.caption(f"✏ {len(revisions)} revision(s) — {rev_summary}")

    # Newsletter format
    if response_type == "newsletter" and result.get("newsletter"):
        st.subheader("Generated Newsletter")
        st.markdown(safe_md(result["newsletter"]["newsletter"]))
        if show_revision_controls:
            revision_controls("newsletter", "Newsletter")

        render_market_section(result, expanded=False, show_controls=show_revision_controls)
        render_performance_section(result, expanded=False, show_controls=show_revision_controls)
        render_risk_section(result, expanded=False, show_controls=show_revision_controls)
        render_stress_test_section(result)

    # other
    elif response_type in ("analysis", "chat"):
        if result.get("market") and not result.get("performance") and not result.get("risk"):
            st.markdown(safe_md(result["market"]["analysis"]))
            if show_revision_controls:
                revision_controls("market", "Market Context")
        else:
            render_market_section(result, expanded=("market_analysis" in result.get("plan", {}).get("intent", "")), show_controls=show_revision_controls)
            render_performance_section(result, expanded=True, show_controls=show_revision_controls)
            render_risk_section(result, expanded=True, show_controls=show_revision_controls)

        render_stress_test_section(result)

    # Fallback failsafe
    else:
        if result.get("newsletter"):
            st.markdown(safe_md(result["newsletter"]["newsletter"]))
            if show_revision_controls:
                revision_controls("newsletter", "Newsletter")
        render_market_section(result, expanded=False, show_controls=show_revision_controls)
        render_performance_section(result, expanded=False, show_controls=show_revision_controls)
        render_risk_section(result, expanded=False, show_controls=show_revision_controls)
        render_stress_test_section(result)

    # One overall Sources footer for every response type
    render_sources(result)

    # Debug critique logs
    if show_revision_controls and st.session_state.get("role") == "admin":
        with st.expander("Debug: Self-Critique Logs", expanded=False):
            for agent_key, lbl in [("performance", "Portfolio Performance"), ("newsletter", "Newsletter")]:
                logs = result.get(agent_key, {}).get("critique_log", [])
                if logs:
                    st.markdown(f"**{lbl}** — {len(logs)} critique round(s)")
                    for i, log in enumerate(logs):
                        st.markdown(f"Round {i+1}: `{log[:200]}{'...' if len(log) > 200 else ''}`")
                else:
                    st.markdown(f"**{lbl}** — no critique log")


def render_validation_agent(hist_entry: dict):
    """
    validation_agent block — single complaint box + version stack + confirm step.
    Sits at the top of the latest response. Additive: the existing per-section
    revise controls remain untouched below.
    """
    result = hist_entry["result"]

    st.markdown(
        '<div class="section-label" style="margin-top:0.5rem;">validation_agent</div>',
        unsafe_allow_html=True,
    )

    # ── Version stack — restore a previous version of this result ──
    versions = hist_entry.get("versions", [])
    if versions:
        col_v, col_r = st.columns([5, 1])
        with col_v:
            labels = [f"{i+1}. {v['label']}" for i, v in enumerate(versions)]
            choice = st.selectbox(
                "Previous versions",
                options=labels,
                key="version_select",
                label_visibility="collapsed",
            )
        with col_r:
            if st.button("Restore", key="btn_restore_version", use_container_width=True):
                idx = labels.index(choice)
                # Restore = new snapshot (so the user gets free redo)
                hist_entry.setdefault("versions", []).append({
                    "label":  "before restore",
                    "result": result,
                })
                hist_entry["result"] = versions[idx]["result"]
                st.rerun()

    # ── Pending plan → confirm step ──
    pending_plan = st.session_state.get("pending_validation_plan")
    if pending_plan:
        st.markdown(
            '<div class="section-label" style="margin-top:0.8rem;">Proposed change</div>',
            unsafe_allow_html=True,
        )
        st.info(pending_plan["summary"])
        col_c, col_x = st.columns([1, 1])
        with col_c:
            if st.button("Confirm", key="btn_validation_confirm", use_container_width=True):
                st.session_state["pending_validation_execute"] = pending_plan
                st.session_state.pop("pending_validation_plan", None)
                st.rerun()
        with col_x:
            if st.button("Cancel", key="btn_validation_cancel", use_container_width=True):
                # Keep the complaint text so the user can edit and re-plan
                st.session_state.pop("pending_validation_plan", None)
                st.rerun()
        return

    # ── Complaint box ──
    complaint = st.text_input(
        "validation_complaint",
        label_visibility="collapsed",
        placeholder="Something off? e.g. 'the Feb numbers look wrong' or 'make the outlook less defensive'",
        key="validation_complaint_text",
    )
    if st.button("Analyze request", key="btn_validation_analyze"):
        if complaint.strip():
            asked_at = wayfound.utc_now_iso()
            with st.spinner("Working out what needs to change..."):
                plan = asyncio.run(validator.plan_revision(complaint.strip(), result))
            # Wayfound: record the complaint + proposed plan (even if later cancelled)
            recorder = hist_entry.get("wayfound")
            if recorder:
                recorder.record(wayfound.messages_for_validation_plan(
                    complaint.strip(), plan, asked_at
                ))
            st.session_state["pending_validation_plan"] = plan
            st.rerun()
        else:
            st.warning("Describe what you'd like changed first.")

    # ── Explanation of the last revision ──
    if result.get("validation_explanation"):
        st.caption(f"✓ {result['validation_explanation']}")
def render_export_controls(result: dict):
    """
    Download the full result as PDF or DOCX. Lazy: nothing is generated until
    the user clicks Prepare, then the two download buttons appear.
    """
    from export import build_pdf, build_docx, export_filename

    st.markdown(
        '<div class="section-label" style="margin-top:0.3rem;">Download</div>',
        unsafe_allow_html=True,
    )
    col_prep, col_pdf, col_docx = st.columns([1, 1, 1])

    with col_prep:
        if st.button("Prepare files", key="btn_prepare_export", use_container_width=True):
            with st.spinner("Generating..."):
                st.session_state["export_pdf"] = build_pdf(result)
                st.session_state["export_docx"] = build_docx(result)
                st.session_state["export_pdf_name"] = export_filename(result, "pdf")
                st.session_state["export_docx_name"] = export_filename(result, "docx")

    if st.session_state.get("export_pdf"):
        with col_pdf:
            st.download_button(
                "PDF",
                data=st.session_state["export_pdf"],
                file_name=st.session_state["export_pdf_name"],
                mime="application/pdf",
                key="dl_pdf",
                use_container_width=True,
            )
        with col_docx:
            st.download_button(
                "DOCX",
                data=st.session_state["export_docx"],
                file_name=st.session_state["export_docx_name"],
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="dl_docx",
                use_container_width=True,
            )


if st.session_state.history:
    latest = st.session_state.history[-1]

    st.divider()
    render_validation_agent(latest)
    render_export_controls(latest["result"])
    render_result(latest["result"], show_revision_controls=True)

    # ── Session history — previous queries this session ────
    if len(st.session_state.history) > 1:
        st.divider()
        st.markdown(
            '<div class="history-label">Previous Queries This Session</div>',
            unsafe_allow_html=True
        )
        for item in reversed(st.session_state.history[:-1]):
            label = item["query"][:80]
            with st.expander(label):
                render_result(item["result"], show_revision_controls=False)