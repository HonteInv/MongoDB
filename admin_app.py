import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
)
from langchain_core.documents import Document
from user_management import create_user, delete_user, list_users, change_role

# ── All data management functions from build_index ─────────
from build import (
    delete_by_source,
    list_sources,
    get_collection,
    embedding,
    COLLECTIONS,
    ingest_pnl_structured,
    delete_pnl_period,
    list_pnl_periods,
)

from auth_helper import verify_login, is_authenticated, is_admin

load_dotenv()

st.set_page_config(
    page_title="RAG Admin",
    page_icon=None,
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
        <a href="https://honte-pnl-query.streamlit.app/" target="_blank">PnL Query</a>
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

    p, .stMarkdown p {
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        color: #4a4a4a !important;
        font-size: 15px !important;
    }

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

    .stFileUploader label {
        font-family: 'Jost', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #7a6e60 !important;
    }

    .stFileUploader > div {
        background-color: #faf8f5 !important;
        border: 1px dashed #ddd5c4 !important;
        border-radius: 2px !important;
    }

    .stSelectbox label {
        font-family: 'Jost', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #7a6e60 !important;
    }

    .stSelectbox > div > div {
        background-color: #faf8f5 !important;
        border: 1px solid #ddd5c4 !important;
        border-radius: 2px !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        color: #2c2c2c !important;
    }

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

    .stButton > button p { color: #f5f0e8 !important; }
    .stButton > button:hover { background-color: #4a4a4a !important; }

    .stSuccess {
        background-color: #f5f5f0 !important;
        border: 1px solid #c9a87a !important;
        border-left: 3px solid #c9a87a !important;
        color: #2c2c2c !important;
        border-radius: 1px !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
    }

    .stInfo {
        background-color: #faf8f5 !important;
        border: 1px solid #ddd5c4 !important;
        border-left: 3px solid #b8a99a !important;
        color: #2c2c2c !important;
        border-radius: 1px !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
    }

    .stWarning {
        background-color: #faf8f5 !important;
        border: 1px solid #ddd5c4 !important;
        color: #7a6e60 !important;
        border-radius: 1px !important;
    }

    .stError {
        background-color: #fdf5f5 !important;
        border: 1px solid #e8c4c4 !important;
        color: #8a4a4a !important;
        border-radius: 1px !important;
    }

    .stSpinner > div { border-top-color: #c9a87a !important; }

    hr {
        border: none !important;
        border-top: 1px solid #e8e2d9 !important;
        margin: 2rem 0 !important;
    }

    .stats-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #e8e2d9;
        font-family: 'Jost', sans-serif;
        font-size: 14px;
        font-weight: 300;
        color: #2c2c2c;
    }
    .stats-label {
        font-weight: 500;
        letter-spacing: 0.04em;
        color: #4a4a4a;
    }
    .stats-count {
        font-family: 'Cormorant Garamond', serif;
        font-size: 18px;
        font-weight: 600;
        color: #c9a87a;
    }
    .source-item {
        font-family: 'Jost', sans-serif;
        font-size: 13px;
        font-weight: 300;
        color: #4a4a4a;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f0ebe3;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Auth — admin role only
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None

if not is_authenticated(st.session_state):
    st.title("Admin View")
    st.divider()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = verify_login(username.strip(), password)
        if user and user["role"] == "admin":
            st.session_state.authenticated = True
            st.session_state.username = user["username"]
            st.session_state.role = user["role"]
            st.rerun()
        elif user:
            st.error("You do not have admin permissions.")
        else:
            st.error("Incorrect username or password.")
    st.stop()

# ============================================================
# Header
# ============================================================

col1, col2 = st.columns([6, 1])
with col1:
    st.title("Admin Panel")
    st.markdown(f"Logged in as **{st.session_state.username}**")
with col2:
    st.markdown("<div style='padding-top: 1.8rem;'></div>", unsafe_allow_html=True)
    if st.button("Logout"):
        for key in ["authenticated", "username", "role"]:
            st.session_state[key] = None
        st.session_state.authenticated = False
        st.rerun()

st.divider()

# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Upload & Index",
    "Delete Documents",
    "Browse Sources",
    "Database Stats",
    "User Management",
    "PnL Periods",
])


# ============================================================
# TAB 1 — Upload & Index
# ============================================================

with tab1:
    st.subheader("Upload & Index Documents")
    st.markdown("Documents indexed here are immediately available to all users.")

    if "upload_category" not in st.session_state:
        st.session_state.upload_category = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    store_type = st.selectbox(
        "Select Knowledge Base",
        list(COLLECTIONS.keys()),
        key="upload_collection",
    )

    if st.session_state.upload_category != store_type:
        st.session_state.upload_category = store_type
        st.session_state.uploader_key += 1 

    uploaded_files = st.file_uploader(
        "Upload Doc (PDF, Markdown, CSV)",
        type=["pdf", "md", "csv"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    collection_name, index_name = COLLECTIONS[store_type]

    if store_type == "pnl":
        st.info(
            "PnL files are stored as structured rows in `pnl_table` (no vector embedding). "
            "Period is auto-detected from the filename — confirm or override below before indexing."
        )
    elif store_type == "context":
        st.info(
            "Context PDFs go through the **daily pipeline**: narrative is date-tagged into "
            "`context_daily`, and the exhibit tables are extracted into `daily_table_data` "
            "(exact numbers via vision). Dates are read from the document content. PDF only."
        )
    else:
        st.info(f"Files will be added to **{collection_name}**.")

    upload_mode = st.radio(
        "Upload mode",
        ["Add new (skip if exists)", "Reindex (replace existing)"],
        help="Reindex deletes existing chunks for this file first, then re-ingests.",
        key="upload_mode",
    )

    # PnL: show period preview + override fields before indexing
    if store_type == "pnl" and uploaded_files:
        from build import _extract_report_period
        st.markdown("**Confirm reporting periods:**")
        period_overrides = {}
        aum_overrides = {}  # {filename: {"start": float|None, "end": float|None}}
        for uf in uploaded_files:
            detected = _extract_report_period(uf.name)
            override = st.text_input(
                f"{uf.name}",
                value=detected,
                help="Format: YYYY-MM. Edit if the auto-detected period is wrong.",
                key=f"period_{uf.name}",
            )
            period_overrides[uf.name] = override.strip()

            # Manual AUM override — shown collapsed by default, use for files
            # that have no AUM summary row (e.g. older 2025 files)
            with st.expander(f"AUM override for {uf.name} (optional)", expanded=False):
                st.caption(
                    "Leave blank to use values parsed from the file. "
                    "Fill in both fields if the file has no AUM summary row, "
                    "or to correct a wrong parsed value."
                )
                col_s, col_e = st.columns(2)
                with col_s:
                    start_val = st.text_input(
                        "Start AUM ($)",
                        value="",
                        placeholder="e.g. 160000000",
                        key=f"start_aum_{uf.name}",
                    )
                with col_e:
                    end_val = st.text_input(
                        "End AUM ($)",
                        value="",
                        placeholder="e.g. 155000000",
                        key=f"end_aum_{uf.name}",
                    )
                aum_overrides[uf.name] = {
                    "start": float(start_val.replace(",", "").replace("$", "")) if start_val.strip() else None,
                    "end":   float(end_val.replace(",", "").replace("$", ""))   if end_val.strip()   else None,
                }

    if uploaded_files and st.button("Index Document", key="btn_upload"):

        # ── PnL: structured row storage (no vector embedding) ──────────
        if store_type == "pnl":
            with st.spinner("Parsing and storing PnL data..."):
                for uploaded_file in uploaded_files:
                    suffix = Path(uploaded_file.name).suffix
                    confirmed_period = period_overrides.get(uploaded_file.name)
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        temp_path = Path(tmp.name)
                    try:
                        if "Reindex" in upload_mode:
                            deleted = delete_pnl_period(confirmed_period)
                            if deleted:
                                st.write(f"Removed {deleted} existing rows for period `{confirmed_period}`")

                        aum_ov = aum_overrides.get(uploaded_file.name, {})
                        n = ingest_pnl_structured(
                            temp_path,
                            report_period=confirmed_period,
                            source_name=uploaded_file.name,
                            uploaded_by=st.session_state.username,
                            start_aum_override=aum_ov.get("start"),
                            end_aum_override=aum_ov.get("end"),
                        )
                        if n:
                            st.success(f"`{uploaded_file.name}` — {n} rows stored as period `{confirmed_period}`")
                        else:
                            st.warning(f"`{uploaded_file.name}` — no rows ingested (period `{confirmed_period}` may already exist)")
                    except Exception as e:
                        st.error(f"`{uploaded_file.name}`: {e}")
                    finally:
                        temp_path.unlink(missing_ok=True)

        # ── Context: daily pipeline (narrative → context_daily, tables → daily_table_data) ──
        elif store_type == "context":
            import daily_data as dd
            import daily_table_data as dtd
            from anthropic import Anthropic

            client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
            reindex = "Reindex" in upload_mode

            with st.spinner("Ingesting daily context (narrative + exact tables via vision)..."):
                for uploaded_file in uploaded_files:
                    if not uploaded_file.name.lower().endswith(".pdf"):
                        st.warning(f"`{uploaded_file.name}` skipped — daily context must be a PDF.")
                        continue

                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        temp_path = Path(tmp.name)

                    try:
                        if reindex:
                            get_collection("context_daily").delete_many({"source": uploaded_file.name})
                            get_collection("daily_table_data").delete_many({"source": uploaded_file.name})

                        # 1) Narrative → context_daily   2) Exact-number tables → daily_table_data (3-pass vision)
                        narr = dd.ingest_file(
                            temp_path, skip_existing=not reindex, source_name=uploaded_file.name
                        )
                        dtd.ingest_tables(
                            temp_path, client, skip_existing=not reindex, source_name=uploaded_file.name
                        )

                        # Concise summary: chunks + days
                        if narr.get("skipped"):
                            st.info(f"`{uploaded_file.name}` — already ingested (skipped)")
                        else:
                            st.success(
                                f"`{uploaded_file.name}` — {narr.get('inserted', 0)} chunks, "
                                f"{len(narr.get('days', []))} day(s)"
                            )

                    except Exception as e:
                        st.error(f"`{uploaded_file.name}`: {e}")
                    finally:
                        temp_path.unlink(missing_ok=True)

        # ── All other categories: vector embedding ──────────────────────
        else:
            all_chunks = []
            errors = []

            with st.spinner("Processing and embedding..."):
                for uploaded_file in uploaded_files:
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        temp_path = Path(tmp.name)

                    try:
                        if "Reindex" in upload_mode:
                            deleted = delete_by_source(uploaded_file.name, collection_name)
                            if deleted:
                                st.write(f"Removed {deleted} existing chunks for `{uploaded_file.name}`")

                        ext = temp_path.suffix.lower()
                        if ext == ".pdf":
                            docs = PyPDFLoader(str(temp_path)).load()
                        elif ext == ".md":
                            text = temp_path.read_text(encoding="utf-8")
                            docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]
                        elif ext == ".csv":
                            docs = CSVLoader(str(temp_path)).load()
                        else:
                            raise ValueError(f"Unsupported file type: {ext}")

                        for doc in docs:
                            doc.metadata = doc.metadata or {}
                            doc.metadata["source"] = uploaded_file.name
                            doc.metadata["original_filename"] = uploaded_file.name
                            doc.metadata["uploaded_by"] = st.session_state.username
                            doc.metadata["collection"] = store_type

                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = splitter.split_documents(docs)
                        all_chunks.extend(chunks)
                        st.write(f"`{uploaded_file.name}` — {len(chunks)} chunks")

                    except Exception as e:
                        errors.append(f"{uploaded_file.name}: {str(e)}")
                        st.warning(f"Error: `{uploaded_file.name}` — {str(e)}")
                    finally:
                        temp_path.unlink(missing_ok=True)

                if all_chunks:
                    MongoDBAtlasVectorSearch.from_documents(
                        documents=all_chunks,
                        embedding=embedding,
                        collection=get_collection(collection_name),
                        index_name=index_name,
                    )
                    st.success(
                        f"{len(all_chunks)} chunks indexed into `{store_type}`. "
                        "All users can access this data immediately."
                    )
                if errors:
                    st.error(f"Failed: {', '.join(errors)}")


# ============================================================
# TAB 2 — Delete Documents
# ============================================================

with tab2:
    st.subheader("Delete Documents")
    st.markdown("Remove all chunks belonging to a specific source file.")

    del_collection = st.selectbox(
        "Knowledge Base",
        list(COLLECTIONS.keys()),
        key="del_collection",
    )
    del_col_name, _ = COLLECTIONS[del_collection]
    
    if st.button("Load Sources", key="btn_load_del_sources"):
        with st.spinner("Loading sources..."):
            sources = list_sources(del_col_name, silent=True)
            source_filenames = sorted(set(Path(s).name for s in sources if s))
            st.session_state["del_sources"] = source_filenames

    source_filenames = st.session_state.get("del_sources", [])

    if source_filenames:
        selected_source = st.selectbox(
            "Select file to delete",
            source_filenames,
            key="del_source",
        )

        dry_run_del = st.checkbox(
            "Preview only (dry run)", value=True, key="del_dryrun"
        )

        if st.button("Delete Selected File", key="btn_delete"):
            with st.spinner("Processing..."):
                count = delete_by_source(selected_source, del_col_name, dry_run=dry_run_del)
                # Context spans two collections — also remove the exact-number tables
                extra = 0
                if del_collection == "context":
                    extra = delete_by_source(
                        selected_source, "daily_table_data", dry_run=dry_run_del
                    )
            targets = f"`{del_col_name}`" + (" + `daily_table_data`" if del_collection == "context" else "")
            if dry_run_del:
                st.info(
                    f"Preview: {count + extra} records would be deleted for "
                    f"`{selected_source}` from {targets}. Uncheck preview to delete."
                )
            else:
                st.success(
                    f"Deleted {count + extra} records for `{selected_source}` from {targets}."
                )
    else:
        st.info("No source files found in this collection.")


# ============================================================
# TAB 3 — Browse Sources
# ============================================================

with tab3:
    st.subheader("Browse Indexed Sources")
    st.markdown("See every source file currently indexed in a collection.")

    browse_cat = st.selectbox(
        "Collection",
        list(COLLECTIONS.keys()),
        key="browse_collection",
    )
    browse_col_name, _ = COLLECTIONS[browse_cat]

    if st.button("Load Sources", key="btn_sources"):
        with st.spinner("Loading..."):
            sources = list_sources(browse_col_name, silent=True)
            filenames = sorted(set(Path(s).name for s in sources if s))

        if filenames:
            st.markdown(
                f"**{len(filenames)} files indexed in `{browse_col_name}`:**"
            )
            for fname in filenames:
                st.markdown(
                    f'<div class="source-item">📄 {fname}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No sources found in this collection.")


# ============================================================
# TAB 4 — Database Stats
# ============================================================

with tab4:
    st.subheader("Database Status")
    st.markdown("Live document counts across all collections.")

    if st.button("Refresh Stats", key="btn_stats"):
        with st.spinner("Fetching..."):
            # Vector collections from COLLECTIONS + the structured ones not in that map
            all_cols = [c for c, _ in COLLECTIONS.values()] + \
                       ["daily_table_data", "pnl_table", "pnl_summary"]
            for col_name in all_cols:
                try:
                    count = get_collection(col_name).count_documents({})
                    st.markdown(
                        f'<div class="stats-row">'
                        f'<span class="stats-label">{col_name}</span>'
                        f'<span class="stats-count">{count:,} documents</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.warning(f"Could not fetch count for {col_name}: {e}")


# ============================================================
# TAB 5 — User Management
# ============================================================
with tab5:
    st.subheader("User Management")

    # ── Create user ────────────────────────────────────────
    st.markdown("**Add New User**")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_username = st.text_input("Username", key="new_username")
    with col2:
        new_password = st.text_input("Password", type="password", key="new_password")
    with col3:
        new_role = st.selectbox("Role", ["guest", "admin"], key="new_role")

    if st.button("Create User", key="btn_create_user"):
        if new_username and new_password:
            try:
                create_user(new_username, new_password, new_role)
                st.success(f"Created {new_role} user: {new_username}")
            except Exception as e:
                st.error(f"Failed: {e}")
        else:
            st.warning("Username and password are required.")

    st.divider()

    # ── List + delete users ────────────────────────────────
    st.markdown("**Current Users**")

    if st.button("Refresh Users", key="btn_refresh_users"):
        client = MongoClient(os.getenv("MONGO_URI_ADMIN"))
        users = list(
            client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["users"]
            .find({}, {"username": 1, "role": 1, "created_at": 1})
        )

        for u in users:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(
                    f'<div class="source-item">{u["username"]}</div>',
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f'<div class="source-item">{u["role"]}</div>',
                    unsafe_allow_html=True
                )
            with col3:
                # Prevent admin from deleting themselves
                if u["username"] != st.session_state.username:
                    if st.button("Remove", key=f"del_user_{u['username']}"):
                        delete_user(u["username"])
                        st.success(f"Removed {u['username']}")
                        st.rerun()


# ============================================================
# TAB 6 — PnL Periods
# ============================================================

with tab6:
    st.subheader("PnL Periods")
    st.markdown(
        "Manage structured PnL data stored in `pnl_table`. "
        "Each period corresponds to one uploaded monthly PnL file."
    )

    if st.button("Refresh Periods", key="btn_pnl_periods"):
        col = get_collection("pnl_table")
        periods = sorted(col.distinct("report_period"))

        if not periods:
            st.info("No PnL periods found in pnl_table.")
        else:
            st.markdown(f"**{len(periods)} period(s) loaded:**")
            for p in periods:
                count = col.count_documents({"report_period": p})
                src_doc = col.find_one({"report_period": p}, {"source_file": 1, "uploaded_by": 1, "uploaded_at": 1})
                src = (src_doc or {}).get("source_file", "?")
                by  = (src_doc or {}).get("uploaded_by", "?")
                st.markdown(
                    f'<div class="stats-row">'
                    f'<span class="stats-label">{p}</span>'
                    f'<span style="font-family:\'Jost\',sans-serif;font-size:13px;color:#7a6e60;">'
                    f'{count} positions · {src} · uploaded by {by}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.divider()
    st.markdown("**Delete a Period**")
    st.markdown("Removes all rows for the selected period so it can be re-uploaded.")

    del_period_input = st.text_input(
        "Period to delete (YYYY-MM)",
        placeholder="e.g. 2026-02",
        key="del_period_input",
    )
    dry_run_period = st.checkbox("Preview only (dry run)", value=True, key="pnl_del_dryrun")

    if st.button("Delete Period", key="btn_del_period"):
        if del_period_input.strip():
            with st.spinner("Processing..."):
                count = delete_pnl_period(del_period_input.strip(), dry_run=dry_run_period)
            if dry_run_period:
                st.info(f"Preview: {count} rows would be deleted for `{del_period_input}`. Uncheck preview to delete.")
            else:
                st.success(f"Deleted {count} rows for period `{del_period_input}`.")
        else:
            st.warning("Enter a period in YYYY-MM format.")

    st.divider()
    st.markdown("**Repair / Set AUM Summary**")
    st.markdown(
        "Directly write a `pnl_summary` document for any period without re-uploading the file. "
        "Use this for older files that have no AUM row, or to correct a wrong value."
    )

    col_rp, col_sa, col_ea = st.columns(3)
    with col_rp:
        repair_period = st.text_input(
            "Period (YYYY-MM)",
            placeholder="e.g. 2025-07",
            key="repair_period",
        )
    with col_sa:
        repair_start = st.text_input(
            "Start AUM ($)",
            placeholder="e.g. 160000000",
            key="repair_start_aum",
        )
    with col_ea:
        repair_end = st.text_input(
            "End AUM ($)",
            placeholder="e.g. 155000000",
            key="repair_end_aum",
        )

    if st.button("Write AUM Summary", key="btn_repair_aum"):
        errors = []
        if not repair_period.strip():
            errors.append("Period is required.")
        if not repair_start.strip():
            errors.append("Start AUM is required.")
        if not repair_end.strip():
            errors.append("End AUM is required.")

        if errors:
            for e in errors:
                st.warning(e)
        else:
            try:
                start_f = float(repair_start.replace(",", "").replace("$", ""))
                end_f   = float(repair_end.replace(",", "").replace("$", ""))
                period  = repair_period.strip()

                # Check the period exists in pnl_table first
                pnl_col = get_collection("pnl_table")
                if pnl_col.count_documents({"report_period": period}) == 0:
                    st.warning(f"No rows found in pnl_table for `{period}`. Make sure the PnL file is uploaded first.")
                else:
                    return_pct = round((end_f - start_f) / start_f * 100, 4) if start_f > 0 else None

                    # Also pull total_pnl from existing pnl_summary if available
                    existing_summary = get_collection("pnl_summary").find_one({"report_period": period}, {"_id": 0})
                    total_pnl = (existing_summary or {}).get("total_pnl")

                    from datetime import datetime, timezone
                    doc = {
                        "report_period": period,
                        "start_aum":     start_f,
                        "end_aum":       end_f,
                        "return_pct":    return_pct,
                        "repaired_by":   st.session_state.username,
                        "repaired_at":   datetime.now(timezone.utc).isoformat(),
                    }
                    if total_pnl is not None:
                        doc["total_pnl"] = total_pnl

                    summary_col = get_collection("pnl_summary")
                    summary_col.delete_many({"report_period": period})
                    summary_col.insert_one(doc)

                    ret_str = f"{return_pct:+.4f}%" if return_pct is not None else "n/a"
                    st.success(
                        f"AUM summary written for `{period}` — "
                        f"start=${start_f:,.0f} · end=${end_f:,.0f} · return={ret_str}"
                    )
            except ValueError:
                st.error("Start AUM and End AUM must be valid numbers.")