# -*- coding: utf-8 -*-
"""
app/streamlit_app.py
--------------------
Interactive Streamlit web interface for the Stylometric AI Text Detector.
Combines RoBERTa semantic embeddings and 16 linguistic stylometric features.
Uses predict_long for chunked analysis of longer texts.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path so 'predict' can be imported from app/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import plotly.graph_objects as go
import pypdf


# =============================================================================
# CSS — kept in a dedicated function
# =============================================================================
def inject_custom_css():
    """Inject all custom CSS styles for the product-page look."""
    st.markdown(
        """
        <style>
        /* ── Google Font ────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

        /* ── Hero Section (tighter bottom gap) ───────────────── */
        .hero {
            text-align: center;
            padding: 1.8rem 1rem 0.4rem;
        }
        .hero h1 {
            font-size: 3rem;
            font-weight: 900;
            background: linear-gradient(135deg, #0fd9b2, #0ea5e9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            letter-spacing: -1px;
        }
        .hero .tagline {
            font-size: 1.15rem;
            color: #8b949e;
            margin-bottom: 0.2rem;
        }
        .hero .caption {
            font-size: 0.82rem;
            color: #484f58;
            letter-spacing: 0.5px;
        }

        /* ── Result Cards (consistent 10px radius) ──────────── */
        .result-card {
            padding: 1.4rem 1.6rem;
            border-radius: 10px;
            margin: 0.6rem 0 1rem;
            border: 1px solid rgba(255,255,255,0.06);
            backdrop-filter: blur(12px);
        }
        .result-human {
            background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(6,95,70,0.08));
            border-left: 5px solid #10b981;
            box-shadow: 0 0 24px rgba(16,185,129,0.10);
        }
        .result-ai {
            background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(127,29,29,0.08));
            border-left: 5px solid #ef4444;
            box-shadow: 0 0 24px rgba(239,68,68,0.10);
        }
        .result-card .verdict-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #8b949e;
            font-weight: 600;
        }
        .result-card .verdict-value {
            font-size: 1.8rem;
            font-weight: 800;
            margin-top: 0.3rem;
        }
        .result-card .confidence {
            font-size: 2.8rem;
            font-weight: 900;
            margin-top: 0.4rem;
            line-height: 1;
        }
        .confidence-human { color: #34d399; }
        .confidence-ai    { color: #f87171; }

        /* ── Empty-state placeholder ─────────────────────────── */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 280px;
            text-align: center;
            color: #484f58;
        }
        .empty-state .empty-icon {
            font-size: 3.2rem;
            margin-bottom: 0.8rem;
            opacity: 0.45;
        }
        .empty-state .empty-text {
            font-size: 0.95rem;
            color: #6e7681;
            line-height: 1.5;
        }
        .empty-state .empty-text strong {
            color: #8b949e;
        }

        /* ── Sidebar Metrics Card (consistent 10px radius) ──── */
        .sidebar-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 0.8rem;
        }
        .sidebar-card h4 {
            margin: 0 0 0.4rem 0;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #0fd9b2;
        }
        .sidebar-card table {
            width: 100%;
            font-size: 0.82rem;
            border-collapse: collapse;
        }
        .sidebar-card td {
            padding: 3px 0;
            color: #c9d1d9;
        }
        .sidebar-card td:last-child {
            text-align: right;
            font-weight: 600;
            color: #e6edf3;
        }

        /* ── Footer ─────────────────────────────────────────── */
        .footer-note {
            font-size: 0.78rem;
            color: #484f58;
            text-align: center;
            padding: 2rem 0 1rem;
            letter-spacing: 0.3px;
        }

        /* ── Buttons — consistent radius, no truncation ─────── */
        .stButton > button {
            border-radius: 10px !important;
            padding: 8px 12px !important;
            font-size: 0.88rem !important;
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: unset !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0fd9b2, #0ea5e9) !important;
            border: none !important;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        /* ── Text area ──────────────────────────────────────── */
        div[data-testid="stTextArea"] textarea {
            border: 1px solid rgba(15,217,178,0.25);
            border-radius: 10px;
            font-size: 0.95rem;
        }
        div[data-testid="stTextArea"] textarea:focus {
            border-color: #0fd9b2;
            box-shadow: 0 0 0 2px rgba(15,217,178,0.15);
        }

        /* ── Tabs ───────────────────────────────────────────── */
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
        }

        /* ── Responsive: narrower viewports ─────────────────── */
        @media (max-width: 768px) {
            .hero h1 { font-size: 2rem; }
            .hero .tagline { font-size: 0.95rem; }
            .result-card .confidence { font-size: 2rem; }
            .result-card .verdict-value { font-size: 1.3rem; }
        }

        /* ── Explanation Section ────────────────────────────── */
        .explain-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.83rem;
            margin: 0.6rem 0 0.8rem;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(22, 27, 34, 0.6);
        }
        .explain-table th {
            background: rgba(255,255,255,0.06);
            color: #8b949e;
            text-align: left;
            padding: 8px 12px;
            font-weight: 600;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        .explain-table td {
            padding: 7px 12px;
            border-top: 1px solid rgba(255,255,255,0.04);
            color: #c9d1d9;
            vertical-align: middle;
        }
        .explain-table tr:hover {
            background: rgba(255,255,255,0.02);
        }
        .explain-table tr.ttr-highlight {
            background: rgba(15, 217, 178, 0.12) !important;
            border-left: 3px solid #0fd9b2;
            font-weight: 600;
        }
        .explain-badge-human {
            display: inline-block;
            background: rgba(16, 185, 129, 0.18);
            color: #34d399;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.74rem;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .explain-badge-ai {
            display: inline-block;
            background: rgba(239, 68, 68, 0.18);
            color: #f87171;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.74rem;
            font-weight: 600;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Plotly gauge / bar helper
# =============================================================================
def build_probability_bar(human_prob: float, ai_prob: float):
    """Return a Plotly horizontal stacked bar for Human vs AI probability."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Probability"],
        x=[human_prob * 100],
        orientation="h",
        name="Human",
        marker_color="#10b981",
        text=f"{human_prob*100:.1f}%",
        textposition="inside",
        textfont=dict(size=14, color="#fff", family="Inter"),
        hovertemplate="Human: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=["Probability"],
        x=[ai_prob * 100],
        orientation="h",
        name="AI",
        marker_color="#ef4444",
        text=f"{ai_prob*100:.1f}%",
        textposition="inside",
        textfont=dict(size=14, color="#fff", family="Inter"),
        hovertemplate="AI: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        height=70,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5,
            font=dict(size=12, color="#8b949e", family="Inter"),
        ),
        showlegend=True,
    )
    return fig


def build_explanation_chart(explanations: list):
    """
    Return a Plotly horizontal bar chart of the top 8 features by abs(pull_ai).
    Red if pulling toward AI (> 0), Green if pulling toward Human (< 0).
    """
    FEATURE_LABELS = {
        "ttr": "Vocab Diversity (ttr)",
        "avg_sent_len": "Avg Sentence Length",
        "std_sent_len": "Sentence Length Std Dev",
        "num_sentences": "Sentence Count",
        "avg_word_len": "Avg Word Length",
        "noun_ratio": "Noun Ratio",
        "verb_ratio": "Verb Ratio",
        "adj_ratio": "Adjective Ratio",
        "adv_ratio": "Adverb Ratio",
        "pron_ratio": "Pronoun Ratio",
        "comma_freq": "Comma Frequency",
        "period_freq": "Period Frequency",
        "question_freq": "Question Frequency",
        "avg_paragraph_len": "Avg Paragraph Length",
        "fk_grade": "Flesch-Kincaid Grade",
        "flesh_ease": "Flesch Reading Ease",
    }

    top8 = explanations[:8]
    # Reverse so top pulling feature appears at the top of the horizontal bar chart
    top8_rev = list(reversed(top8))

    y_labels = [FEATURE_LABELS.get(x["feature"], x["feature"]) for x in top8_rev]
    x_vals = [x["pull_ai"] for x in top8_rev]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in x_vals]

    hover_texts = [
        f"<b>{lbl}</b><br>Pull: {v:+.2f} (→ {'AI' if v > 0 else 'Human'})<br>Raw Value: {x['value']}<br>Closer to: {x['closer_to']}"
        for lbl, v, x in zip(y_labels, x_vals, top8_rev)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y_labels,
        x=x_vals,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hoverinfo="text",
        hovertext=hover_texts,
        text=[f"{v:+.2f}" for v in x_vals],
        textposition="outside",
        textfont=dict(size=11, color="#8b949e", family="Inter"),
    ))

    max_abs = max([abs(v) for v in x_vals] + [0.5]) * 1.35

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=30, t=10, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[-max_abs, max_abs],
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.25)",
            zerolinewidth=1.5,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(size=10, color="#8b949e", family="Inter"),
            title=dict(
                text="← Pulls toward Human   |   Pulls toward AI →",
                font=dict(size=11, color="#8b949e", family="Inter"),
            ),
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#e6edf3", family="Inter"),
            showgrid=False,
        ),
        showlegend=False,
    )
    return fig


# =============================================================================
# Page Config & CSS (must be first Streamlit command)
# =============================================================================
st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.caption("MODEL STATUS")

    st.markdown(
        """
        <div class="sidebar-card">
            <h4>Model</h4>
            <p style="color:#e6edf3; font-size:0.9rem; margin:0;">
                Hybrid RoBERTa‑base + 16 stylometric features
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <h4>Hybrid Model Metrics</h4>
            <table>
                <tr><td>Accuracy</td><td>96.79 %</td></tr>
                <tr><td>Precision</td><td>96.14 %</td></tr>
                <tr><td>Recall</td><td>97.48 %</td></tr>
                <tr><td>F1-Score</td><td>96.81 %</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <h4>Baseline (RoBERTa only)</h4>
            <table>
                <tr><td>Accuracy</td><td>95.47 %</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <h4>How It Works</h4>
            <p style="color:#c9d1d9; font-size:0.82rem; margin:0; line-height:1.55;">
                Text is encoded with RoBERTa to capture semantic meaning, then
                augmented with 16 handcrafted stylometric features (readability,
                POS-tag ratios, punctuation frequency, sentence-length variance,
                and more). A meta-classifier fuses both signals for the final
                Human vs AI verdict. Texts longer than ~150 words are split
                into chunks at sentence boundaries and results are
                word-count-weighted.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Cached Resource Loader (Load Model & Scaler Once)
# =============================================================================
@st.cache_resource(show_spinner="⏳ Loading models, first run takes about 45 seconds...")
def load_prediction_pipeline():
    """
    Loads and caches the prediction pipeline and scaler once.
    Re-uses cached instances across reruns and button clicks.
    """
    from predict_long import predict_long
    from predict import scaler
    return predict_long, scaler


# =============================================================================
# Hero Section
# =============================================================================
st.markdown(
    """
    <div class="hero">
        <h1>AI Text Detector</h1>
        <div class="tagline">Detect AI-generated text with state-of-the-art hybrid deep learning.</div>
        <div class="caption">Hybrid RoBERTa + 16 stylometric features</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Sample texts for quick demonstration
# =============================================================================
SAMPLE_HUMAN = (
    "honestly idk why people keep hyping up that new restaurant downtown lol. "
    "me and my buddy waited like 45 mins in the freezing rain just for soggy fries "
    "and a burger that tasted completely burnt. totally not worth the 30 bucks!! "
    "save your cash and just grab some tacos down the street instead imo."
)

SAMPLE_AI = (
    "When you use your computer, it stores files on your hard drive. "
    "Sometimes, when you save a file, your computer might not have enough space next to the other files "
    "to store the whole file in one piece. Instead, it has to save part of the file in one place "
    "and the rest in another place. This is called fragmentation. A disk defragmenter fixes this by "
    "reorganizing those pieces back together so your computer can locate them much faster and run smoothly."
)

if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""


def set_human_sample():
    st.session_state["user_text"] = SAMPLE_HUMAN


def set_ai_sample():
    st.session_state["user_text"] = SAMPLE_AI


# =============================================================================
# Tabs
# =============================================================================
tab_analyse, tab_comparison, tab_about = st.tabs(
    ["🔬 Analyse Text", "📊 Model Comparison", "ℹ️ About"]
)

# ── Helper: PDF text extraction ─────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file using pypdf."""
    try:
        reader = pypdf.PdfReader(uploaded_file)
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)
        return "\n".join(pages_text).strip()
    except Exception as e:
        st.error(f"Failed to read PDF file: {e}")
        return ""


# ── Helper: Unified Prediction & Explanation Renderer ───────────────────────
def render_prediction_results(text: str):
    """
    Run predict_long on text and render the verdict card, Plotly probability bar,
    per-chunk breakdown, and full stylometric explanation panel.
    """
    try:
        predict_long_fn, scaler = load_prediction_pipeline()

        with st.spinner("Analyzing linguistic patterns and semantics..."):
            res = predict_long_fn(text)
            if len(res) == 4:
                label, probs, chunk_ai_probs, raw_features = res
            else:
                label, probs, chunk_ai_probs = res
                raw_features = probs.get("raw_features", {})

        human_prob = float(probs.get("Human-written", 0.0))
        ai_prob = float(probs.get("AI-generated", 0.0))

        # Verdict card
        if label == "Human-written":
            card_cls = "result-human"
            conf_cls = "confidence-human"
            icon = "👤"
            conf_pct = human_prob * 100
        else:
            card_cls = "result-ai"
            conf_cls = "confidence-ai"
            icon = "🤖"
            conf_pct = ai_prob * 100

        st.markdown(
            f"""
            <div class="result-card {card_cls}">
                <div class="verdict-label">Overall Verdict</div>
                <div class="verdict-value">{icon} {label}</div>
                <div class="confidence {conf_cls}">{conf_pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Plotly probability bar
        st.plotly_chart(
            build_probability_bar(human_prob, ai_prob),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Per-chunk breakdown
        if len(chunk_ai_probs) > 1:
            with st.expander(f"📊 Chunk breakdown ({len(chunk_ai_probs)} chunks)"):
                for i, cp in enumerate(chunk_ai_probs, 1):
                    chunk_label = "AI" if cp > 0.5 else "Human"
                    st.progress(cp, text=f"Chunk {i}: AI {cp*100:.2f}% — {chunk_label}")

        # ── Explanation Section ─────────────────────────────────────
        st.markdown("---")
        st.markdown("##### 🧬 Stylometric Feature Explanation")

        from explain import explain_features
        explanations = explain_features(raw_features, scaler)

        # Top 8 horizontal bar chart
        st.plotly_chart(
            build_explanation_chart(explanations),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Top 8 features table (ensuring ttr is included for highlighting)
        top_features = list(explanations[:8])
        if not any(f["feature"] == "ttr" for f in top_features):
            ttr_entry = next((f for f in explanations if f["feature"] == "ttr"), None)
            if ttr_entry:
                top_features.append(ttr_entry)

        FEATURE_LABELS = {
            "ttr": "Vocabulary Diversity (ttr)",
            "avg_sent_len": "Avg Sentence Length",
            "std_sent_len": "Sentence Length Std Dev",
            "num_sentences": "Sentence Count",
            "avg_word_len": "Avg Word Length",
            "noun_ratio": "Noun Ratio",
            "verb_ratio": "Verb Ratio",
            "adj_ratio": "Adjective Ratio",
            "adv_ratio": "Adverb Ratio",
            "pron_ratio": "Pronoun Ratio",
            "comma_freq": "Comma Frequency",
            "period_freq": "Period Frequency",
            "question_freq": "Question Frequency",
            "avg_paragraph_len": "Avg Paragraph Length",
            "fk_grade": "Flesch-Kincaid Grade",
            "flesh_ease": "Flesch Reading Ease",
        }

        table_rows = []
        for f in top_features:
            is_ttr = f["feature"] == "ttr"
            row_cls = ' class="ttr-highlight"' if is_ttr else ""
            badge_cls = "explain-badge-human" if f["closer_to"] == "Human" else "explain-badge-ai"
            icon = "👤" if f["closer_to"] == "Human" else "🤖"
            fname = FEATURE_LABELS.get(f["feature"], f["feature"])
            if is_ttr:
                fname += " ⭐ <i>(Strongest Discriminator, Cohen's d = 1.53)</i>"

            raw_v = f"{f['value']:.4f}" if isinstance(f['value'], float) else str(f['value'])

            table_rows.append(
                f"<tr{row_cls}>"
                f"<td>{fname}</td>"
                f"<td><code>{raw_v}</code></td>"
                f"<td><span class=\"{badge_cls}\">{icon} {f['closer_to']}</span></td>"
                f"</tr>"
            )

        table_html = f"""
        <table class="explain-table">
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Input Raw Value</th>
                    <th>Closer To</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        st.caption(
            "This shows which writing-style features pushed the prediction toward Human or AI, "
            "based on comparison to class averages. It is a heuristic explanation, not an exact "
            "attribution of the transformer's decision."
        )

    except Exception as e:
        st.error(f"❌ Prediction failed with an error: {e}")


# ── Tab 1: Analyse Text ─────────────────────────────────────────────────────
with tab_analyse:
    subtab_paste, subtab_pdf = st.tabs(["✍️ Paste Text", "📄 Upload PDF"])

    # ── Subtab 1: Paste Text ────────────────────────────────────────────────
    with subtab_paste:
        col_input, col_result = st.columns([1.1, 1], gap="large")

        with col_input:
            st.markdown("#### ✍️ Input Text")

            btn_col1, btn_col2 = st.columns(2)
            btn_col1.button("👤 Human Sample", on_click=set_human_sample, use_container_width=True)
            btn_col2.button("🤖 AI Sample", on_click=set_ai_sample, use_container_width=True)

            input_text = st.text_area(
                "Enter or paste the text you want to evaluate:",
                key="user_text",
                height=220,
                placeholder="Paste at least 20 words here...",
                help="Excerpts with 20 or more words provide reliable stylometric and grammatical signals.",
            )

            words_in_text = [w for w in input_text.strip().split() if w]
            word_count = len(words_in_text)
            st.caption(f"Word count: **{word_count}** (minimum 20 required)")

            analyse_clicked = st.button("🚀 Analyse", type="primary", key="btn_analyse_text", use_container_width=True)

        with col_result:
            st.markdown("#### 📋 Results")

            if analyse_clicked:
                if word_count < 20:
                    st.warning(
                        f"⚠️ Text is too short for reliable analysis ({word_count} words). "
                        "Please provide at least 20 words to proceed."
                    )
                else:
                    render_prediction_results(input_text)
            else:
                st.markdown(
                    '<div class="empty-state">'
                    '  <div class="empty-icon">🔬</div>'
                    '  <div class="empty-text">'
                    '    Paste your text on the left and click<br>'
                    '    <strong>Analyse</strong> to see the verdict here.'
                    '  </div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

    # ── Subtab 2: Upload PDF ────────────────────────────────────────────────
    with subtab_pdf:
        col_pdf_input, col_pdf_result = st.columns([1.1, 1], gap="large")

        with col_pdf_input:
            st.markdown("#### 📄 Upload PDF Document")
            uploaded_pdf = st.file_uploader(
                "Upload a PDF file to evaluate:",
                type=["pdf"],
                help="Upload a text-based PDF file (.pdf only).",
                key="pdf_uploader",
            )

            extracted_pdf_text = ""
            pdf_word_count = 0

            if uploaded_pdf is not None:
                extracted_pdf_text = extract_text_from_pdf(uploaded_pdf)
                pdf_words = [w for w in extracted_pdf_text.split() if w]
                pdf_word_count = len(pdf_words)

                # Show preview (first ~300 chars) in an expander
                preview_snippet = extracted_pdf_text[:300]
                if len(extracted_pdf_text) > 300:
                    preview_snippet += "..."

                with st.expander("📄 PDF Preview (first ~300 chars)", expanded=True):
                    st.write(preview_snippet if preview_snippet else "*(No readable text found)*")

                st.caption(f"Extracted word count: **{pdf_word_count}** words")

        with col_pdf_result:
            st.markdown("#### 📋 Results")

            if uploaded_pdf is not None:
                if pdf_word_count < 20:
                    st.warning("No readable text found — this may be a scanned PDF.")
                else:
                    render_prediction_results(extracted_pdf_text)
            else:
                st.markdown(
                    '<div class="empty-state">'
                    '  <div class="empty-icon">📄</div>'
                    '  <div class="empty-text">'
                    '    Upload a PDF document on the left<br>'
                    '    to see the verdict here.'
                    '  </div>'
                    '</div>',
                    unsafe_allow_html=True,
                )


# ── Tab 2: Model Comparison ─────────────────────────────────────────────────
with tab_comparison:
    st.markdown("#### 📊 Model Comparison")
    st.markdown(
        "Visual comparison of the **Hybrid** (RoBERTa + stylometric) model "
        "versus the **Baseline** (RoBERTa-only) model."
    )

    plots_dir = ROOT_DIR / "analysis" / "plots"

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        cm_hybrid = plots_dir / "confusion_matrix_hybrid.png"
        if cm_hybrid.exists():
            st.image(str(cm_hybrid), caption="Hybrid Model — Confusion Matrix", use_container_width=True)
    with comp_col2:
        cm_baseline = plots_dir / "confusion_matrix_baseline.png"
        if cm_baseline.exists():
            st.image(str(cm_baseline), caption="Baseline Model — Confusion Matrix", use_container_width=True)

    bar_chart = plots_dir / "model_comparison_bar.png"
    if bar_chart.exists():
        st.image(str(bar_chart), caption="Metric Comparison — Hybrid vs Baseline", use_container_width=True)

# ── Tab 3: About ────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("#### ℹ️ About This Project")
    st.markdown(
        """
        The **AI Text Detector** is a hybrid classifier that fuses
        RoBERTa‑base contextual embeddings with 16 handcrafted stylometric
        features to distinguish human-written text from AI-generated text.

        It was trained and evaluated on the **HC3** (Human ChatGPT Comparison)
        corpus.
        """
    )

    st.markdown("#### ⚠️ Limitations")
    st.warning(
        "**Results are probabilistic, not definitive.**\n\n"
        "Accuracy may decrease in the following scenarios:\n\n"
        "- **Very short texts** (< 50 words) — limited stylometric signal.\n"
        "- **Heavily edited text** — human post-editing of AI output can shift features.\n"
        "- **Long formal text** — academic or legal prose may exhibit patterns similar to AI-generated text.\n\n"
        "Always use this tool as a supporting signal, not as sole evidence."
    )

    st.markdown("#### 🏗️ Architecture")
    st.markdown(
        """
        1. **RoBERTa‑base** encodes the text into a 768-d semantic vector.
        2. **16 stylometric features** are extracted (readability indices,
           POS-tag ratios, punctuation frequencies, sentence-length statistics, etc.).
        3. Both representations are concatenated (784-d) and passed to a
           **Logistic Regression** meta-classifier.
        4. Texts longer than ~150 words are chunked at sentence boundaries;
           chunk predictions are word-count-weighted.
        """
    )


# =============================================================================
# Footer
# =============================================================================
st.markdown(
    '<div class="footer-note">Stylometric Detection System · HC3 Corpus · '
    'RoBERTa + 16 Handcrafted Features</div>',
    unsafe_allow_html=True,
)
