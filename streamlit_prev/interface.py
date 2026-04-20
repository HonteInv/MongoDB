import streamlit as st
from alex_agent import build_agent

st.set_page_config(
    page_title="HonTe",
    page_icon="",
    layout="wide"
)

# -------------------------
# Top Navigation Bar
# -------------------------

st.markdown("""
    <style>
        header[data-testid="stHeader"] {
            display: none !important;
        }
 
        .navbar {
            display: flex;
            align-items: center;
            background-color: #f5f0e8;
            padding: 14px 32px;
            margin: -60px -4rem 20px -4rem;
            border-bottom: 1px solid #ddd5c4;
            gap: 32px;
        }
        .navbar-brand {
            font-family: 'Cormorant Garamond', serif;
            font-weight: 600;
            font-size: 17px;
            color: #2c2c2c;
            text-decoration: none;
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
        .navbar a:hover {
            color: #2c2c2c;
        }
    </style>

    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Jost:wght@300;400;500&display=swap" rel="stylesheet">

    <div class="navbar">
        <span class="navbar-brand">Navigation</span>
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

    .main {
        background-color: #ffffff;
    }

    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
    }

    /* Caption / subtitle */
    .stCaption {
        color: #9a8f82 !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        font-size: 14px !important;
        letter-spacing: 0.03em !important;
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

    .stButton > button:hover {
        background-color: #4a4a4a !important;
    }

    /* Answer block */
    .answer-block {
        background-color: #faf8f5;
        border-left: 2px solid #c9a87a;
        padding: 2rem 2.5rem;
        margin-top: 1.5rem;
        border-radius: 0;
        font-size: 16px;
        line-height: 1.9;
        color: #2c2c2c;
        font-family: 'Jost', sans-serif;
        font-weight: 300;
    }

    /* Source tags */
    .source-tag {
        display: inline-block;
        background-color: #f5f0e8;
        color: #7a6e60;
        font-size: 11px;
        padding: 3px 10px;
        border-radius: 1px;
        margin: 3px;
        font-family: 'Jost', monospace;
        letter-spacing: 0.04em;
        border: 1px solid #e0d8cc;
    }

    /* Divider */
    .divider {
        border: none;
        border-top: 1px solid #e8e2d9;
        margin: 2rem 0;
    }

    /* Warning */
    .stAlert {
        background-color: #faf8f5 !important;
        border: 1px solid #ddd5c4 !important;
        color: #7a6e60 !important;
        border-radius: 1px !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #c9a87a !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Load Agent
# ============================================================

@st.cache_resource
def get_agent():
    return build_agent()

try:
    agent = get_agent()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ============================================================
# UI
# ============================================================

st.title("HonTe")
st.caption("Ask anything. Answers drawn from the CEO's books, transcripts, and interviews.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

query = st.text_area(
    "Your Question",
    height=120,
    placeholder="What is the CEO's view on long-term thinking vs short-term performance?",
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("Ask")

# ============================================================
# Response
# ============================================================

if ask and query.strip():
    with st.spinner("Thinking..."):
        result = agent.query(query)

    st.markdown(
        f'<div class="answer-block">{result["answer"]}</div>',
        unsafe_allow_html=True
    )

    # Show unique sources
    unique_sources = list(set(result["sources"]))
    if unique_sources:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("**Sources referenced:**")
        source_html = " ".join(f'<span class="source-tag">{s}</span>' for s in unique_sources)
        st.markdown(source_html, unsafe_allow_html=True)

elif ask:
    st.warning("Please enter a question.")