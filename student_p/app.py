"""
app.py
Streamlit Web Application — Student Placement Predictor
Author: Project by Smrity Shreya
"""

import pickle
import numpy as np
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Card container */
    .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
    }

    /* Header */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        letter-spacing: -0.5px;
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        text-align: center;
        color: rgba(255,255,255,0.55);
        font-size: 1rem;
        margin-bottom: 0;
    }
    .badge {
        display: inline-block;
        background: linear-gradient(90deg, #6c63ff, #48cfad);
        color: white;
        border-radius: 50px;
        padding: 4px 16px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* Section headers */
    .section-header {
        color: #a78bfa;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    /* Slider / widget labels */
    label {
        color: rgba(255,255,255,0.85) !important;
        font-weight: 600 !important;
    }

    /* Predict button */
    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, #6c63ff 0%, #48cfad 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: 0 8px 24px rgba(108,99,255,0.4);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(108,99,255,0.6);
    }

    /* Metric chips */
    .metric-chip {
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-chip .val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #a78bfa;
    }
    .metric-chip .lbl {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.5);
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Override Streamlit success/warning */
    div[data-testid="stAlert"] {
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load Artefacts ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artefacts():
    with open("placement_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artefacts()
    artefacts_ok = True
except FileNotFoundError:
    artefacts_ok = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<div style='text-align:center'><span class='badge'>🤖 ML Classifier · Random Forest</span></div>", unsafe_allow_html=True)
st.markdown("<div class='hero-title'>🎓 Student Placement<br>Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Enter your academic profile to predict campus placement probability</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if not artefacts_ok:
    st.error(
        "⚠️  Model files not found.\n\n"
        "Run `generate_dataset.py` then `eda_and_modeling.py` first to create "
        "`placement_model.pkl` and `scaler.pkl`, then restart this app."
    )
    st.stop()

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📊 Academic Profile</div>", unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.slider(
            "CGPA",
            min_value=5.0,
            max_value=10.0,
            value=7.5,
            step=0.1,
            help="Your Cumulative Grade Point Average on a 10-point scale.",
        )
        internships = st.slider(
            "Internships Completed",
            min_value=0,
            max_value=3,
            value=1,
            step=1,
            help="Number of internships done (0–3).",
        )
        projects = st.slider(
            "Projects Completed",
            min_value=0,
            max_value=4,
            value=2,
            step=1,
            help="Number of academic / personal projects (0–4).",
        )

    with col2:
        tech_score = st.slider(
            "Technical Skills Score",
            min_value=50,
            max_value=100,
            value=75,
            step=1,
            help="Self-assessed or test-based technical skill score (50–100).",
        )
        backlogs = st.selectbox(
            "Active Core Backlogs",
            options=[0, 1, 2],
            index=0,
            help="Number of currently active core subject backlogs.",
            format_func=lambda x: f"{x} backlog{'s' if x != 1 else ''}",
        )

# ── Live Profile Summary ──────────────────────────────────────────────────────
st.markdown("<br><div class='section-header'>📋 Your Profile Summary</div>", unsafe_allow_html=True)

cols = st.columns(5)
metrics = [
    ("CGPA", f"{cgpa:.1f}"),
    ("Internships", str(internships)),
    ("Projects", str(projects)),
    ("Tech Score", str(tech_score)),
    ("Backlogs", str(backlogs)),
]
for col, (lbl, val) in zip(cols, metrics):
    col.markdown(
        f"<div class='metric-chip'>"
        f"<div class='val'>{val}</div>"
        f"<div class='lbl'>{lbl}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍  Predict Placement Status")

if predict_clicked:
    input_data = np.array([[cgpa, internships, projects, tech_score, backlogs]])
    input_scaled = scaler.transform(input_data)

    prediction   = model.predict(input_scaled)[0]
    proba        = model.predict_proba(input_scaled)[0]
    placed_prob  = proba[1] * 100
    not_placed_prob = proba[0] * 100

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🎯 Prediction Result</div>", unsafe_allow_html=True)

    if prediction == 1:
        st.success(
            f"🎉 **Congratulations! You are likely to be PLACED!**\n\n"
            f"Our model predicts a **{placed_prob:.1f}%** placement probability based on your profile. "
            f"Keep polishing your skills and ace those interviews — you're on the right track! 🚀"
        )
    else:
        st.warning(
            f"💪 **Keep Going — You've Got This!**\n\n"
            f"The model currently estimates a **{placed_prob:.1f}%** placement chance. "
            f"Focus on boosting your CGPA, adding internships, and building more projects. "
            f"Every effort brings you closer to your dream job! 🌟"
        )

    # Probability bar
    st.markdown("<br>", unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    pc1.metric("✅ Placement Probability", f"{placed_prob:.1f}%")
    pc2.metric("❌ Non-Placement Probability", f"{not_placed_prob:.1f}%")

    st.progress(int(placed_prob))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:rgba(255,255,255,0.25); font-size:0.78rem;'>"
    "Built by <strong>Smrity Shreya</strong> · B.Tech CSE 3rd Year · "
    "Random Forest Classifier · scikit-learn · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
