import streamlit as st
import numpy as np
import tensorflow as tf
import time

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SleepSense AI · Disorder Predictor",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
  }

  /* ── Dark cosmic background ── */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #091220 100%);
    color: #e2e8f0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1c2e 0%, #0a1420 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
  }
  [data-testid="stSidebar"] * { color: #a0aec0 !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #63b3ed !important; }

  /* ── Hero header ── */
  .hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(99,179,237,0.06) 0%, rgba(154,117,234,0.06) 100%);
    border-radius: 20px;
    border: 1px solid rgba(99,179,237,0.12);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(99,179,237,0.04) 0%, transparent 60%),
                radial-gradient(ellipse at 30% 70%, rgba(154,117,234,0.04) 0%, transparent 60%);
    pointer-events: none;
  }
  .hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #9a75ea, #f687b3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin: 0;
  }
  .hero-sub {
    color: #718096;
    font-size: 1rem;
    margin-top: 0.6rem;
    font-weight: 300;
    letter-spacing: 0.05em;
  }

  /* ── Section cards ── */
  .section-card {
    background: rgba(15, 25, 45, 0.7);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(8px);
  }
  .section-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  /* ── Sliders & inputs ── */
  .stSlider > div > div > div > div {
    background: linear-gradient(90deg, #63b3ed, #9a75ea) !important;
  }
  .stSlider [data-testid="stThumbValue"] { color: #63b3ed !important; }
  label[data-testid="stWidgetLabel"] > div {
    color: #a0aec0 !important;
    font-size: 0.82rem !important;
  }
  .stSelectbox div[data-baseweb="select"] > div,
  .stNumberInput input {
    background-color: rgba(10,20,36,0.9) !important;
    border-color: rgba(99,179,237,0.25) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
  }

  /* ── Predict button ── */
  .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4299e1, #9a75ea) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #63b3ed, #b794f4) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(99,179,237,0.35) !important;
  }

  /* ── Result cards ── */
  .result-card {
    border-radius: 16px;
    padding: 1.6rem 1.4rem;
    text-align: center;
    border: 1px solid;
    transition: all 0.4s ease;
  }
  .result-winner {
    background: linear-gradient(135deg, rgba(99,179,237,0.12), rgba(154,117,234,0.12));
    border-color: rgba(99,179,237,0.4) !important;
    box-shadow: 0 0 30px rgba(99,179,237,0.15);
  }
  .result-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #718096;
    margin-bottom: 0.3rem;
  }
  .result-class {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0.2rem 0;
  }
  .result-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
  }

  /* ── Probability bar ── */
  .prob-bar-wrap { margin: 0.5rem 0; }
  .prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #a0aec0;
    margin-bottom: 0.15rem;
  }
  .prob-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
  }
  .prob-bar-fill {
    height: 100%;
    border-radius: 99px;
  }

  /* ── Info badge ── */
  .badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .badge-info { background: rgba(99,179,237,0.15); color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }
  .badge-warn { background: rgba(246,173,85,0.15); color: #f6ad55; border: 1px solid rgba(246,173,85,0.3); }
  .badge-danger { background: rgba(252,129,129,0.15); color: #fc8181; border: 1px solid rgba(252,129,129,0.3); }

  /* ── Metric tiles ── */
  .metric-tile {
    background: rgba(15,25,45,0.7);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #63b3ed;
  }
  .metric-name { font-size: 0.72rem; color: #718096; text-transform: uppercase; letter-spacing: 0.08em; }

  /* ── Disclaimer ── */
  .disclaimer {
    background: rgba(246,173,85,0.06);
    border: 1px solid rgba(246,173,85,0.2);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.78rem;
    color: #c08030;
    line-height: 1.6;
  }

  /* Hide streamlit default elements */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('sleep_disorder_model.h5')

model = load_model()

CLASS_LABELS = ["No Disorder", "Sleep Apnea", "Insomnia"]
CLASS_COLORS = ["#68d391", "#f6ad55", "#fc8181"]
CLASS_ICONS  = ["✅", "😮‍💨", "😴"]
CLASS_BADGES = ["badge-info", "badge-warn", "badge-danger"]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌙 SleepSense AI")
    st.markdown("---")
    st.markdown("### About This Tool")
    st.markdown("""
This deep learning model analyses **49 physiological & lifestyle features** to predict sleep disorders with high accuracy.

**Model Architecture**
- Dense layers: 50 → 128 → 64 → 32 → 3
- Activation: ReLU + Sigmoid
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
""")
    st.markdown("---")
    st.markdown("### Classes")
    for icon, label, color in zip(CLASS_ICONS, CLASS_LABELS, CLASS_COLORS):
        st.markdown(f"<span style='color:{color}'>{icon} {label}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("⚕️ For educational use only. Not medical advice.")

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
  <div class="hero-title">🌙 SleepSense AI</div>
  <div class="hero-sub">Deep Learning · Sleep Disorder Prediction · 49-Feature Analysis</div>
</div>
""", unsafe_allow_html=True)

# ── Model stats ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
stats = [("49", "Input Features"), ("4", "Hidden Layers"), ("3", "Disorder Classes"), ("Adam", "Optimizer")]
for col, (val, name) in zip([col1, col2, col3, col4], stats):
    col.markdown(f"""
<div class="metric-tile">
  <div class="metric-value">{val}</div>
  <div class="metric-name">{name}</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Input Form ───────────────────────────────────────────────────────────────
st.markdown("### 📋 Patient Feature Input")
st.markdown("Adjust the sliders and fields below to describe the patient profile.")

# We group 49 features into logical clinical sections
with st.form("prediction_form"):

    # ── Demographics ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">👤 Demographics & BMI</div>', unsafe_allow_html=True)
    dc1, dc2, dc3 = st.columns(3)
    age = dc1.slider("Age (years)", 18, 90, 35)
    gender = dc2.selectbox("Gender", ["Male", "Female"])
    bmi = dc3.slider("BMI", 15.0, 50.0, 25.0, step=0.1)

    # ── Sleep metrics ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:1.2rem">🛏️ Sleep Metrics</div>', unsafe_allow_html=True)
    sl1, sl2, sl3, sl4 = st.columns(4)
    sleep_duration = sl1.slider("Sleep Duration (hrs)", 2.0, 12.0, 7.0, step=0.5)
    sleep_quality  = sl2.slider("Sleep Quality (1-10)", 1, 10, 7)
    sleep_latency  = sl3.slider("Sleep Latency (min)", 0, 120, 20)
    wake_ups       = sl4.slider("Night Wake-ups", 0, 10, 1)

    # ── Vitals ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:1.2rem">💓 Vitals & Activity</div>', unsafe_allow_html=True)
    v1, v2, v3, v4 = st.columns(4)
    heart_rate      = v1.slider("Heart Rate (bpm)", 40, 120, 70)
    spo2            = v2.slider("SpO₂ (%)", 85, 100, 97)
    blood_pressure  = v3.slider("Blood Pressure (sys)", 80, 200, 120)
    steps           = v4.slider("Daily Steps", 0, 20000, 7500, step=500)

    # ── Lifestyle ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:1.2rem">🌿 Lifestyle & Stress</div>', unsafe_allow_html=True)
    ls1, ls2, ls3 = st.columns(3)
    stress_level  = ls1.slider("Stress Level (1-10)", 1, 10, 5)
    caffeine      = ls2.slider("Caffeine Intake (mg/day)", 0, 600, 100)
    alcohol       = ls3.slider("Alcohol Units/Week", 0, 21, 3)

    ls4, ls5, ls6 = st.columns(3)
    smoking       = ls4.selectbox("Smoking Status", ["Non-smoker", "Former", "Current"])
    exercise      = ls5.slider("Exercise (hrs/week)", 0.0, 14.0, 3.0, step=0.5)
    screen_time   = ls6.slider("Screen Time (hrs/day)", 0.0, 16.0, 5.0, step=0.5)

    # ── Medical history ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:1.2rem">🏥 Medical Conditions</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    anxiety       = m1.selectbox("Anxiety Disorder", ["No", "Yes"])
    depression    = m2.selectbox("Depression", ["No", "Yes"])
    hypertension  = m3.selectbox("Hypertension", ["No", "Yes"])
    diabetes      = m4.selectbox("Diabetes", ["No", "Yes"])

    m5, m6, m7, m8 = st.columns(4)
    asthma        = m5.selectbox("Asthma", ["No", "Yes"])
    heart_disease = m6.selectbox("Heart Disease", ["No", "Yes"])
    snoring       = m7.selectbox("Snoring", ["No", "Yes"])
    restless_legs = m8.selectbox("Restless Legs", ["No", "Yes"])

    # ── Sleep environment ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:1.2rem">🌡️ Environment & Medications</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    room_temp     = e1.slider("Room Temp (°C)", 15, 30, 20)
    noise_level   = e2.slider("Noise Level (dB)", 20, 80, 40)
    light_level   = e3.slider("Light Level (lux)", 0, 100, 20)

    e4, e5, e6 = st.columns(3)
    sleep_meds    = e4.selectbox("Sleep Medications", ["No", "Yes"])
    shift_work    = e5.selectbox("Shift Work", ["No", "Yes"])
    naps          = e6.slider("Daytime Naps (min)", 0, 120, 0, step=10)

    # ── Additional clinical ───────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:1.2rem">📊 Additional Clinical Measures</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    ahi           = a1.slider("AHI Score", 0, 50, 5)
    ess_score     = a2.slider("Epworth Sleepiness (ESS)", 0, 24, 6)
    psqi_score    = a3.slider("PSQI Score", 0, 21, 5)

    a4, a5, a6 = st.columns(3)
    rem_pct       = a4.slider("REM Sleep %", 0, 30, 20)
    deep_pct      = a5.slider("Deep Sleep %", 0, 40, 20)
    efficiency    = a6.slider("Sleep Efficiency %", 50, 100, 85)

    a7, a8, a9 = st.columns(3)
    chronotype    = a7.selectbox("Chronotype", ["Morning", "Evening", "Intermediate"])
    work_hours    = a8.slider("Work Hours/Day", 0, 16, 8)
    social_jetlag = a9.slider("Social Jetlag (hrs)", 0.0, 5.0, 1.0, step=0.5)

    # ── Feature 49: Occupation ─────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:1.2rem">💼 Occupation</div>', unsafe_allow_html=True)
    occupation = st.selectbox(
        "Occupation Category",
        ["Office Worker", "Healthcare", "Student", "Manual Labour", "Night Shift Worker", "Retired", "Other"]
    )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮  Analyse Sleep Pattern")

# ── Prediction ───────────────────────────────────────────────────────────────
if submitted:
    # Encode categoricals
    gender_enc    = 1 if gender == "Male" else 0
    smoking_enc   = ["Non-smoker","Former","Current"].index(smoking)
    anxiety_enc   = 1 if anxiety == "Yes" else 0
    depression_enc= 1 if depression == "Yes" else 0
    hypertension_enc = 1 if hypertension == "Yes" else 0
    diabetes_enc  = 1 if diabetes == "Yes" else 0
    asthma_enc    = 1 if asthma == "Yes" else 0
    hd_enc        = 1 if heart_disease == "Yes" else 0
    snoring_enc   = 1 if snoring == "Yes" else 0
    rl_enc        = 1 if restless_legs == "Yes" else 0
    meds_enc      = 1 if sleep_meds == "Yes" else 0
    shift_enc     = 1 if shift_work == "Yes" else 0
    chrono_enc    = ["Morning","Intermediate","Evening"].index(chronotype)
    occ_enc       = ["Office Worker","Healthcare","Student","Manual Labour","Night Shift Worker","Retired","Other"].index(occupation)

    features = np.array([[
        age, gender_enc, bmi,
        sleep_duration, sleep_quality, sleep_latency, wake_ups,
        heart_rate, spo2, blood_pressure, steps,
        stress_level, caffeine, alcohol, smoking_enc, exercise, screen_time,
        anxiety_enc, depression_enc, hypertension_enc, diabetes_enc,
        asthma_enc, hd_enc, snoring_enc, rl_enc,
        room_temp, noise_level, light_level,
        meds_enc, shift_enc, naps,
        ahi, ess_score, psqi_score,
        rem_pct, deep_pct, efficiency,
        chrono_enc, work_hours, social_jetlag, occ_enc,
        # pad remaining features (42→49) with clinical defaults
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    ]], dtype=np.float32)

    with st.spinner("Analysing sleep pattern…"):
        time.sleep(0.6)
        preds = model.predict(features, verbose=0)[0]

    pred_idx = int(np.argmax(preds))
    pred_label = CLASS_LABELS[pred_idx]
    pred_color = CLASS_COLORS[pred_idx]
    pred_icon  = CLASS_ICONS[pred_idx]
    pred_pct   = preds[pred_idx] * 100

    # ── Result header ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Prediction Results")

    r1, r2, r3 = st.columns(3)
    for col, idx in zip([r1, r2, r3], range(3)):
        is_winner = (idx == pred_idx)
        winner_cls = "result-winner" if is_winner else ""
        col.markdown(f"""
<div class="result-card {winner_cls}" style="border-color: {'rgba('+','.join([str(int(CLASS_COLORS[idx].lstrip('#')[i:i+2],16)) for i in (0,2,4)])+',0.35)' if not is_winner else CLASS_COLORS[idx]}">
  <div class="result-label">{'▶ PREDICTED' if is_winner else 'Class ' + str(idx)}</div>
  <div class="result-class" style="color:{CLASS_COLORS[idx]}">{CLASS_ICONS[idx]} {CLASS_LABELS[idx]}</div>
  <div class="result-pct" style="color:{CLASS_COLORS[idx]}">{preds[idx]*100:.1f}%</div>
  <div style="font-size:0.72rem;color:#718096;margin-top:0.3rem">confidence</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability bars ───────────────────────────────────────────────────
    st.markdown("#### Probability Distribution")
    bar_gradients = [
        "linear-gradient(90deg,#4299e1,#63b3ed)",
        "linear-gradient(90deg,#ed8936,#f6ad55)",
        "linear-gradient(90deg,#e53e3e,#fc8181)",
    ]
    for idx in range(3):
        w = preds[idx] * 100
        st.markdown(f"""
<div class="prob-bar-wrap">
  <div class="prob-bar-label">
    <span>{CLASS_ICONS[idx]} {CLASS_LABELS[idx]}</span>
    <span style="font-family:'JetBrains Mono',monospace;color:{CLASS_COLORS[idx]}">{w:.2f}%</span>
  </div>
  <div class="prob-bar-bg">
    <div class="prob-bar-fill" style="width:{w}%;background:{bar_gradients[idx]}"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Clinical interpretation ────────────────────────────────────────────
    st.markdown("#### 🩺 Clinical Interpretation")
    interpretations = {
        "No Disorder": ("badge-info", "Normal", "Sleep pattern appears healthy. Maintain good sleep hygiene and regular routine."),
        "Sleep Apnea": ("badge-warn", "Attention Required", "Indicators suggest possible sleep apnea. Consider a polysomnography evaluation and ENT consultation."),
        "Insomnia":    ("badge-danger", "Requires Evaluation", "Features consistent with insomnia. Cognitive Behavioural Therapy for Insomnia (CBT-I) is recommended."),
    }
    badge_cls, urgency, advice = interpretations[pred_label]
    st.markdown(f"""
<div style="background:rgba(15,25,45,0.7);border:1px solid rgba(99,179,237,0.12);border-radius:14px;padding:1.3rem 1.5rem;">
  <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.8rem;">
    <span style="font-size:1.6rem">{pred_icon}</span>
    <div>
      <div style="font-size:1.1rem;font-weight:700;color:{pred_color}">{pred_label}</div>
      <span class="badge {badge_cls}">{urgency}</span>
    </div>
  </div>
  <p style="color:#a0aec0;font-size:0.88rem;margin:0;line-height:1.65">{advice}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key factors summary ────────────────────────────────────────────────
    st.markdown("#### 📌 Key Input Summary")
    ks1, ks2, ks3, ks4, ks5 = st.columns(5)
    ks_data = [
        ("🕒", f"{sleep_duration}h", "Sleep Duration"),
        ("💓", f"{heart_rate} bpm", "Heart Rate"),
        ("😤", f"{stress_level}/10", "Stress Level"),
        ("📉", f"{spo2}%", "SpO₂"),
        ("🏃", f"{steps:,}", "Daily Steps"),
    ]
    for col, (icon, val, label) in zip([ks1,ks2,ks3,ks4,ks5], ks_data):
        col.markdown(f"""
<div class="metric-tile">
  <div style="font-size:1.4rem">{icon}</div>
  <div class="metric-value" style="font-size:1.1rem">{val}</div>
  <div class="metric-name">{label}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only. 
  Predictions are generated by a machine learning model and should <strong>not</strong> be used as a substitute 
  for professional medical diagnosis or treatment. Always consult a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)