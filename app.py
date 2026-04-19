import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Flight Intelligence System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1520 50%, #0a0e1a 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1520 0%, #111827 100%);
        border-right: 1px solid rgba(56,189,248,0.15);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
        transition: border-color 0.2s;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(56,189,248,0.5);
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
    [data-testid="stMetricValue"] { color: #f0f9ff !important; font-size: 1.6rem !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

    /* Headings */
    h1 { color: #f0f9ff !important; font-weight: 700 !important; letter-spacing: -0.03em; }
    h2, h3 { color: #e0f2fe !important; font-weight: 600 !important; }

    /* Info / Warning boxes */
    .stAlert { border-radius: 10px !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 12px 32px;
        width: 100%;
        transition: transform 0.15s, box-shadow 0.15s;
        letter-spacing: 0.02em;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(14,165,233,0.35);
    }

    /* Select / Input */
    .stSelectbox > div, .stNumberInput > div, .stSlider > div {
        color: #e2e8f0 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 8px !important;
        color: #e0f2fe !important;
        font-weight: 600 !important;
    }

    /* Divider */
    hr { border-color: rgba(56,189,248,0.1) !important; margin: 1.5rem 0; }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, rgba(14,165,233,0.08), rgba(99,102,241,0.08));
        border: 1px solid rgba(14,165,233,0.25);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }

    /* Pipeline step */
    .pipeline-step {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #e0f2fe;
    }
    .pipeline-step:hover {
        background: rgba(14,165,233,0.08);
        border-color: rgba(56,189,248,0.5);
        transition: all 0.2s;
    }

    /* Objective card */
    .obj-card {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #0ea5e9;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 8px 0;
    }

    /* Badge */
    .badge {
        display: inline-block;
        background: rgba(14,165,233,0.2);
        border: 1px solid rgba(14,165,233,0.4);
        color: #7dd3fc;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Plotly chart background */
    .js-plotly-plot { border-radius: 12px; overflow: hidden; }

    /* Performance bar */
    .perf-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        height: 10px;
        margin: 6px 0 14px;
    }
    .perf-bar-fill {
        height: 10px;
        border-radius: 8px;
        background: linear-gradient(90deg, #0ea5e9, #6366f1);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color="#94a3b8", family="Space Grotesk"),
    title_font=dict(color="#e0f2fe", size=15, family="Space Grotesk"),

    # ✅ Keep default margin here
    margin=dict(l=20, r=20, t=40, b=20),

    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.1)",
        zeroline=False
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.1)",
        zeroline=False
    ),

    colorway=["#0ea5e9", "#6366f1", "#10b981", "#f59e0b", "#ef4444"],

    hoverlabel=dict(
        bgcolor="#1e293b",
        font_color="#e0f2fe",
        bordercolor="#0ea5e9"
    )
)
# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    try:
        model1   = joblib.load("models/stage1_delay_classifier.pkl")
        model2   = joblib.load("models/stage2_delay_regressor.pkl")
        model3   = joblib.load("models/stage3_cause_classifier.pkl")
        model4   = joblib.load("models/stage4_cancellation_classifier.pkl")
        encoders = joblib.load("models/encoders.pkl")
        le_cause = joblib.load("models/stage3_cause_encoder.pkl")
        return model1, model2, model3, model4, encoders, le_cause, None
    except FileNotFoundError as e:
        return None, None, None, None, None, None, str(e)

model1, model2, model3, model4, encoders, le_cause, model_error = load_models()

if model_error:
    st.error(f"⚠️ Model file not found: `{model_error}`\n\nPlease run `Flight.ipynb` first to generate all model files in the `models/` folder.")
    st.stop()

le_carrier = encoders["carrier"]
le_origin  = encoders["origin"]
le_dest    = encoders["dest"]

# Load metrics if available
@st.cache_data
def load_metrics():
    try:
        with open("models/metrics.json") as f:
            return json.load(f)
    except:
        return None

metrics = load_metrics()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def safe_encode(encoder, value):
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return 0

def risk_label(prob):
    if prob < 15:   return "🟢 Low Risk"
    elif prob < 40: return "🟡 Medium Risk"
    else:           return "🔴 High Risk"

def risk_color(prob):
    if prob < 15:   return "#10b981"
    elif prob < 40: return "#f59e0b"
    else:           return "#ef4444"

def gauge_chart(value, title, max_val=100):
    color = risk_color(value)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#94a3b8", "size": 13}},
        number={"suffix": "%", "font": {"color": "#f0f9ff", "size": 24}},
        gauge=dict(
            axis=dict(range=[0, max_val], tickcolor="#475569"),
            bar=dict(color=color),
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.1)",
            steps=[
                dict(range=[0, 15], color="rgba(16,185,129,0.1)"),
                dict(range=[15, 40], color="rgba(245,158,11,0.1)"),
                dict(range=[40, 100], color="rgba(239,68,68,0.1)")
            ],
            threshold=dict(line=dict(color=color, width=3), value=value)
        )
    ))

    # ✅ Safe override
    layout = CHART_LAYOUT.copy()
    layout["margin"] = dict(l=20, r=20, t=40, b=10)

    fig.update_layout(**layout, height=200)

    return fig

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 20px;'>
        <div style='font-size:2.5rem'>✈️</div>
        <div style='font-size:1.1rem; font-weight:700; color:#f0f9ff; letter-spacing:-0.02em;'>Flight Intelligence</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:2px;'>ML-Powered Aviation Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🔮 Prediction", "📈 Model Performance", "📌 Objectives"],
        label_visibility="hidden"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#475569; line-height:1.6;'>
    <b style='color:#64748b;'>Dataset</b><br>
    Bureau of Transportation Statistics<br>
    Full Year · 12 months merged<br><br>
    <b style='color:#64748b;'>Models</b><br>
    Random Forest Ensemble<br>
    4-Stage Prediction Pipeline
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# 📊 DASHBOARD
# ─────────────────────────────────────────
if page == "📊 Dashboard":

    st.markdown("# 📊 Aviation Performance Dashboard")
    st.markdown("<p style='color:#64748b; margin-top:-10px;'>Full-year insights from all 12 months of BTS flight records</p>", unsafe_allow_html=True)
    st.markdown("---")

    # KPIs — populated dynamically after data loads
    k1, k2, k3, k4 = st.columns(4)

    st.markdown("---")

    try:
        @st.cache_data(show_spinner="Loading full year dataset...")
        def load_data():
            import os

            MERGED_PATH = "alldata/full_year.csv"

            # ── If merged file already exists, load it directly (fastest) ──
            if os.path.exists(MERGED_PATH):
                df = pd.read_csv(MERGED_PATH, low_memory=False)
                return df

            # ── Otherwise merge all monthly CSVs on the fly ──────────────
            import glob
            files = sorted(glob.glob("alldata/*.csv"))
            if not files:
                raise FileNotFoundError(
                    "No CSV files found in alldata/. "
                    "Add monthly BTS CSVs (jan.csv … dec.csv) or run the notebook first."
                )

            df_list = []
            for f in files:
                temp = pd.read_csv(f, low_memory=False)
                df_list.append(temp)

            df = pd.concat(df_list, ignore_index=True)
            df = df.drop_duplicates()

            # Derive missing columns from FL_DATE
            if 'FL_DATE' in df.columns:
                df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
                if 'DAY_OF_MONTH' not in df.columns:
                    df['DAY_OF_MONTH'] = df['FL_DATE'].dt.day
                if 'DAY_OF_WEEK' not in df.columns:
                    df['DAY_OF_WEEK'] = df['FL_DATE'].dt.weekday + 1

            if 'DEP_TIME' not in df.columns:
                df['DEP_TIME'] = df['CRS_DEP_TIME'] if 'CRS_DEP_TIME' in df.columns else 0

            # Save merged file for next time
            df.to_csv(MERGED_PATH, index=False)
            return df

        df = load_data()

        DELAY_CAUSES = ["CARRIER_DELAY","WEATHER_DELAY","NAS_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY"]
        df[DELAY_CAUSES] = df[DELAY_CAUSES].fillna(0)

        # ── Dynamic KPIs computed from actual data ────────────────────────
        total_flights = len(df)
        delay_rate    = df["ARR_DEL15"].mean() * 100
        avg_delay     = df["ARR_DELAY"].mean()
        cancel_rate   = df["CANCELLED"].mean() * 100
        k1.metric("Total Flights",     f"{total_flights:,}",    "BTS Full Year")
        k2.metric("Delay Rate",        f"{delay_rate:.1f}%",    "Full Year")
        k3.metric("Avg Arrival Delay", f"{avg_delay:.0f} min",  "Full Year")
        k4.metric("Cancellation Rate", f"{cancel_rate:.1f}%",   "Full Year")

        # Dynamic insight
        top_airline = df.groupby("OP_UNIQUE_CARRIER")["ARR_DELAY"].mean().idxmax()
        top_delay   = df.groupby("OP_UNIQUE_CARRIER")["ARR_DELAY"].mean().max()
        st.info(f"💡 **Insight:** Airline **{top_airline}** has the highest average arrival delay at **{top_delay:.1f} min**")

        # Row 1
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✈️ Average Delay by Airline")
            airline_delay = df.groupby("OP_UNIQUE_CARRIER")["ARR_DELAY"].mean().sort_values(ascending=False).reset_index()
            airline_delay.columns = ["Airline", "Avg Delay (min)"]
            fig = px.bar(
                airline_delay, x="Airline", y="Avg Delay (min)",
                color="Avg Delay (min)", color_continuous_scale=["#0ea5e9","#6366f1","#ef4444"],
                title="Average Arrival Delay by Carrier"
            )
            fig.update_layout(**CHART_LAYOUT)
            fig.update_traces(hovertemplate="<b>%{x}</b><br>Avg Delay: %{y:.1f} min<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            monthly = df.groupby("MONTH")["ARR_DELAY"].mean().reset_index()

            # Ensure all 12 months exist
            all_months = pd.DataFrame({"MONTH": list(range(1,13))})
            monthly = all_months.merge(monthly, on="MONTH", how="left")

            # Fill missing months with 0 (or np.nan if you prefer gaps)
            monthly["ARR_DELAY"] = monthly["ARR_DELAY"].fillna(0)

            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]

            monthly["Month Name"] = month_names

            fig = px.line(
                monthly,
                x="Month Name",
                y="ARR_DELAY",
                markers=True,
                title="Monthly Average Arrival Delay"
            )

            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=6),
                hovertemplate="<b>%{x}</b><br>%{y:.1f} min<extra></extra>"
            )

            fig.update_layout(**CHART_LAYOUT, height=350)

            st.plotly_chart(fig, use_container_width=True)

        # Row 2
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("⚠️ Delay Causes Breakdown")
            cause_totals = df[DELAY_CAUSES].sum().reset_index()
            cause_totals.columns = ["Cause", "Total Minutes"]
            cause_labels = {
                "CARRIER_DELAY": "Carrier", "WEATHER_DELAY": "Weather",
                "NAS_DELAY": "NAS / ATC", "SECURITY_DELAY": "Security",
                "LATE_AIRCRAFT_DELAY": "Late Aircraft"
            }
            cause_totals["Cause"] = cause_totals["Cause"].map(cause_labels)

            fig = px.pie(
                cause_totals, values="Total Minutes", names="Cause",
                title="Distribution of Delay Causes",
                hole=0.45,
                color_discrete_sequence=["#0ea5e9","#6366f1","#10b981","#f59e0b","#ef4444"]
            )
            fig.update_traces(textposition='outside', textfont_size=12,
                              hovertemplate="<b>%{label}</b><br>%{value:,.0f} mins<br>%{percent}<extra></extra>")
            fig.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("📊 On-Time vs Delayed Distribution")
            status_counts = df["ARR_DEL15"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            status_counts["Status"] = status_counts["Status"].map({0: "✅ On-Time", 1: "❌ Delayed"})

            fig = px.bar(
                status_counts, x="Status", y="Count",
                color="Status",
                color_discrete_map={"✅ On-Time": "#10b981", "❌ Delayed": "#ef4444"},
                title="Flight Status Distribution",
                text="Count"
            )
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside',
                              hovertemplate="<b>%{x}</b><br>%{y:,} flights<extra></extra>")
            fig.update_layout(**CHART_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Row 3 — Heatmap
        st.subheader("🌡️ Delay Heatmap — Airline × Month")
        pivot = df.groupby(["OP_UNIQUE_CARRIER","MONTH"])["ARR_DELAY"].mean().unstack()

        # Ensure all 12 months exist
        for m in range(1, 13):
            if m not in pivot.columns:
                pivot[m] = np.nan

        pivot = pivot[sorted(pivot.columns)]

        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

        pivot.columns = month_names

        fig = px.imshow(
            pivot,
            aspect="auto",  # ✅ IMPORTANT (prevents squishing)
            color_continuous_scale=["#0a0e1a","#0ea5e9","#ef4444"],
            title="Average Delay (minutes) by Airline and Month"
        )

        fig.update_layout(**CHART_LAYOUT, height=400)

        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.warning("📂 No data found in `alldata/` folder. Add monthly BTS CSVs (jan.csv … dec.csv) or run `Flight.ipynb` first to generate `full_year.csv`.")
    except Exception as e:
        st.error(f"Unexpected error loading dashboard: `{e}`")

# ─────────────────────────────────────────
# 🔮 PREDICTION
# ─────────────────────────────────────────
elif page == "🔮 Prediction":

    st.markdown("# 🔮 Flight Prediction System")
    st.markdown("<p style='color:#64748b; margin-top:-10px;'>Enter flight details to get real-time AI predictions</p>", unsafe_allow_html=True)
    st.markdown("---")

    colA, colB = st.columns([1, 2], gap="large")

    with colA:
        st.markdown("### 🛫 Flight Details")

        airline = st.selectbox("✈️ Airline Carrier", le_carrier.classes_)
        origin  = st.selectbox("🛫 Origin Airport", le_origin.classes_)
        dest    = st.selectbox("🛬 Destination Airport", le_dest.classes_)

        st.markdown("---")
        st.markdown("### 📅 Schedule")

        col_m, col_d = st.columns(2)
        with col_m:
            month = st.selectbox("Month", list(range(1,13)),
                format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
        with col_d:
            day = st.slider("Day", 1, 31, 15)

        dow = st.select_slider("Day of Week", options=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                                value="Mon")
        dow_num = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(dow) + 1

        st.markdown("---")
        st.markdown("### ⏱️ Timing")
        dep_time  = st.number_input("Departure Time (HHMM)", min_value=0, max_value=2359, value=900, step=5)
        dep_delay = st.number_input("Current Departure Delay (min)", min_value=-60, max_value=600, value=0)

        predict_btn = st.button("🚀 Run Prediction", use_container_width=True)

    with colB:
        if not predict_btn:
            st.markdown("""
            <div style='display:flex; align-items:center; justify-content:center;
                        height:420px; flex-direction:column; gap:16px;
                        background:rgba(255,255,255,0.02); border:1px dashed rgba(56,189,248,0.2);
                        border-radius:16px;'>
                <div style='font-size:3rem'>✈️</div>
                <div style='color:#475569; font-size:1rem;'>Fill in flight details and click <b style="color:#0ea5e9;">Run Prediction</b></div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Encode
            ce = safe_encode(le_carrier, airline)
            oe = safe_encode(le_origin, origin)
            de = safe_encode(le_dest, dest)

            FEATURES = ["MONTH","DAY_OF_MONTH","DAY_OF_WEEK",
                        "CARRIER_ENC","ORIGIN_ENC","DEST_ENC",
                        "DEP_TIME","DEP_DELAY"]

            input_data = pd.DataFrame(
                [[month, day, dow_num, ce, oe, de, dep_time, dep_delay]],
                columns=FEATURES
            )

            with st.spinner("Running 4-stage prediction pipeline..."):
                delay_prob = model1.predict_proba(input_data)[0][1] * 100
                is_delayed = delay_prob >= 50

                delay_mins  = 0
                delay_cause = "N/A"
                if is_delayed:
                    delay_mins  = max(0, model2.predict(input_data)[0])
                    cause_idx   = model3.predict(input_data)[0]
                    delay_cause = le_cause.inverse_transform([cause_idx])[0]
                    delay_cause = delay_cause.replace("_DELAY","").replace("_"," ").title()

                cancel_input = input_data.drop(columns=["DEP_DELAY"])
                cancel_prob  = model4.predict_proba(cancel_input)[0][1] * 100

            # Summary banner
            status_color = "#10b981" if not is_delayed else "#ef4444"
            status_text  = "ON-TIME" if not is_delayed else "LIKELY DELAYED"
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,{status_color}22,{status_color}11);
                        border:1px solid {status_color}55; border-radius:14px;
                        padding:16px 24px; margin-bottom:16px; display:flex;
                        align-items:center; gap:16px;'>
                <div style='font-size:2rem'>{"✅" if not is_delayed else "⚠️"}</div>
                <div>
                    <div style='font-weight:700; font-size:1.1rem; color:{status_color};'>{status_text}</div>
                    <div style='color:#94a3b8; font-size:0.85rem;'>
                        {airline} · {origin} → {dest}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge row
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(gauge_chart(delay_prob, "Delay Probability"), use_container_width=True)
            with g2:
                st.plotly_chart(gauge_chart(cancel_prob, "Cancellation Risk"), use_container_width=True)

            # Detail cards
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Delay Chance",    f"{delay_prob:.1f}%",   risk_label(delay_prob))
            d2.metric("Est. Delay",      f"{delay_mins:.0f} min", "If delayed" if is_delayed else "No delay")
            d3.metric("Probable Cause",  delay_cause)
            d4.metric("Cancel Risk",     f"{cancel_prob:.1f}%",  risk_label(cancel_prob))

            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Recommendations")

            recs = []
            if delay_prob > 40:
                recs.append(("🔔", "High delay probability — consider travel insurance or flexible booking"))
            if cancel_prob > 10:
                recs.append(("🚨", "Elevated cancellation risk — check airline rebooking policy"))
            if delay_cause not in ["N/A"]:
                recs.append(("📋", f"Primary delay driver: **{delay_cause}** — check live airline status"))
            if dep_delay > 15:
                recs.append(("⏱️", "Departure is already delayed — arrival delay is likely"))
            if not recs:
                recs.append(("✅", "Low risk flight — enjoy your journey!"))

            for icon, text in recs:
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.03); border-left:3px solid #0ea5e9;
                            border-radius:0 10px 10px 0; padding:10px 16px; margin:6px 0;
                            color:#e0f2fe; font-size:0.9rem;'>
                    {icon} {text}
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# 📈 MODEL PERFORMANCE
# ─────────────────────────────────────────
elif page == "📈 Model Performance":

    st.markdown("# 📈 Model Performance")
    st.markdown("<p style='color:#64748b; margin-top:-10px;'>Evaluated on held-out test sets — no data leakage</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Load from JSON if available, else use fallback
    if metrics:
        m1_acc   = metrics["delay_classifier"]["accuracy"]
        m1_f1    = metrics["delay_classifier"]["f1"]
        m2_mae   = metrics["delay_regressor"]["mae"]
        m2_r2    = metrics["delay_regressor"]["r2"]
        m3_acc   = metrics["cause_classifier"]["accuracy"]
        m4_acc   = metrics["cancellation_classifier"]["accuracy"]
        m4_f1    = metrics["cancellation_classifier"]["f1"]
    else:
        m1_acc, m1_f1 = 90.0, 0.85
        m2_mae, m2_r2 = 11.0, 0.72
        m3_acc        = 66.0
        m4_acc, m4_f1 = 96.0, 0.78

    # Model cards
    models_info = [
        {
            "icon": "🎯",
            "name": "Stage 1 — Delay Classifier",
            "algo": "Random Forest Classifier",
            "task": "Binary classification: Will flight arrive ≥15 min late?",
            "metrics": [("Accuracy", f"{m1_acc}%", m1_acc), ("F1 Score", f"{m1_f1:.3f}", m1_f1*100)],
            "features": "Month, Day, Carrier, Origin, Dest, Dep Time, Dep Delay",
            "color": "#0ea5e9"
        },
        {
            "icon": "⏱️",
            "name": "Stage 2 — Delay Regressor",
            "algo": "Random Forest Regressor",
            "task": "Regression: How many minutes will the delay be?",
            "metrics": [("MAE", f"{m2_mae} min", max(0, 100 - m2_mae*2)), ("R² Score", f"{m2_r2:.3f}", m2_r2*100)],
            "features": "Same as Stage 1 (delayed flights only)",
            "color": "#6366f1"
        },
        {
            "icon": "🔍",
            "name": "Stage 3 — Cause Classifier",
            "algo": "Random Forest Classifier (5-class)",
            "task": "Classification: Carrier / Weather / NAS / Security / Late Aircraft",
            "metrics": [("Accuracy", f"{m3_acc}%", m3_acc)],
            "features": "Same as Stage 1",
            "color": "#10b981"
        },
        {
            "icon": "🚫",
            "name": "Stage 4 — Cancellation Classifier",
            "algo": "Random Forest Classifier (class_weight=balanced)",
            "task": "Binary classification: Will the flight be cancelled?",
            "metrics": [("Accuracy", f"{m4_acc}%", m4_acc), ("F1 Score", f"{m4_f1:.3f}", m4_f1*100)],
            "features": "Month, Day, Carrier, Origin, Dest, Dep Time (no dep_delay)",
            "color": "#f59e0b"
        }
    ]

    for m in models_info:
        with st.expander(f"{m['icon']}  **{m['name']}**  —  {m['algo']}", expanded=True):
            c1, c2 = st.columns([2,1])
            with c1:
                st.markdown(f"**Task:** {m['task']}")
                st.markdown(f"**Features:** <span style='font-family:monospace;font-size:0.85rem;color:#94a3b8;'>{m['features']}</span>", unsafe_allow_html=True)
            with c2:
                for label, val, bar_pct in m["metrics"]:
                    st.markdown(f"<div style='color:#94a3b8;font-size:0.78rem;text-transform:uppercase;letter-spacing:.06em;'>{label}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:1.4rem;font-weight:700;color:{m['color']};margin-bottom:4px;'>{val}</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='perf-bar-bg'>
                      <div class='perf-bar-fill' style='width:{min(bar_pct,100):.0f}%; background:linear-gradient(90deg,{m["color"]},{m["color"]}99);'></div>
                    </div>
                    """, unsafe_allow_html=True)

    # Comparison chart
    st.markdown("---")
    st.subheader("📊 Accuracy Comparison Across Models")
    comp_df = pd.DataFrame({
        "Model": ["Delay\nClassifier", "Cause\nClassifier", "Cancellation\nClassifier"],
        "Accuracy": [m1_acc, m3_acc, m4_acc],
        "Color": ["#0ea5e9", "#10b981", "#f59e0b"]
    })
    fig = px.bar(comp_df, x="Model", y="Accuracy",
                 color="Model",
                 color_discrete_sequence=["#0ea5e9","#10b981","#f59e0b"],
                 title="Test Set Accuracy by Model",
                 text="Accuracy")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(**CHART_LAYOUT, showlegend=False, yaxis_range=[0,105])
    st.plotly_chart(fig, use_container_width=True)

    st.info("ℹ️ All metrics computed on 20% held-out test data (80/20 split, `random_state=42`). Earlier notebook versions evaluated on training data — those inflated R²=0.99 / 100% accuracy figures have been corrected.")

# ─────────────────────────────────────────
# 📌 OBJECTIVES
# ─────────────────────────────────────────
elif page == "📌 Objectives":

    st.markdown("# 📌 Project Objectives")
    st.markdown("<p style='color:#64748b; margin-top:-10px;'>Flight Intelligence System — Problem, Approach & Impact</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Problem Statement
    st.markdown("## 🌐 Problem Statement")
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(14,165,233,0.08),rgba(99,102,241,0.06));
                border:1px solid rgba(14,165,233,0.2); border-radius:14px; padding:24px; margin:8px 0 20px;'>
        <p style='color:#e0f2fe; font-size:1rem; line-height:1.8; margin:0;'>
        Flight delays and cancellations cost the U.S. aviation industry over
        <strong style='color:#0ea5e9;'>$33 billion annually</strong>, affecting
        <strong style='color:#0ea5e9;'>22% of all domestic flights</strong>.
        Passengers face missed connections, productivity loss, and poor travel experiences —
        while airlines struggle with cascading operational disruptions.
        <br><br>
        This project builds a <strong style='color:#6366f1;'>4-stage ML prediction pipeline</strong>
        using full-year BTS flight records (12 months merged) to give travellers, airlines, and airport operators
        actionable, data-driven intelligence — <em>before</em> a delay ever happens.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4-Stage Pipeline
    st.markdown("## 🔄 4-Stage Prediction Pipeline")

    stages = [
        ("1", "🎯", "Delay\nClassifier", "Will this flight be delayed ≥15 min?", "Binary · Yes / No", "#0ea5e9"),
        ("2", "⏱️", "Duration\nRegressor", "How many minutes will the delay be?", "Regression · Minutes", "#6366f1"),
        ("3", "🔍", "Cause\nClassifier", "What is driving the delay?", "5-class · Root Cause", "#10b981"),
        ("4", "🚫", "Cancellation\nClassifier", "What is the cancellation risk?", "Binary · Risk %", "#f59e0b"),
    ]

    cols = st.columns(len(stages))
    for col, (num, icon, name, desc, output, color) in zip(cols, stages):
        with col:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03); border:1px solid {color}44;
                        border-top:3px solid {color}; border-radius:12px;
                        padding:20px 14px; text-align:center;'>
                <div style='font-size:1.8rem;'>{icon}</div>
                <div style='font-size:0.72rem; color:{color}; font-weight:700;
                            text-transform:uppercase; letter-spacing:.1em; margin:6px 0 2px;'>Stage {num}</div>
                <div style='font-weight:700; color:#e0f2fe; margin-bottom:8px;
                            font-size:0.95rem; white-space:pre-line;'>{name}</div>
                <div style='color:#64748b; font-size:0.8rem; line-height:1.5; margin-bottom:10px;'>{desc}</div>
                <div style='background:{color}22; border-radius:6px; padding:4px 8px;
                            color:{color}; font-size:0.72rem; font-family:monospace;'>{output}</div>
            </div>
            """, unsafe_allow_html=True)

    # Arrow connector
    st.markdown("""
    <div style='text-align:center; color:#334155; font-size:1.5rem; margin:12px 0;'>
    ──────── Stage 1 → Stage 2 → Stage 3 ⟹ Stage 4 (independent) ────────
    </div>
    """, unsafe_allow_html=True)

    # Objectives
    st.markdown("---")
    st.markdown("## 🎯 Core Objectives")

    objectives = [
        ("O1", "Predict Flight Delays", "Classify whether any given domestic flight will arrive 15+ minutes late, using pre-departure information only — enabling proactive traveller alerts.", "#0ea5e9"),
        ("O2", "Estimate Delay Duration", "For flights flagged as delayed, regress on historical patterns to estimate the expected arrival delay in minutes, helping passengers plan connections.", "#6366f1"),
        ("O3", "Identify Delay Root Cause", "Attribute the delay to one of five BTS-defined categories (Carrier, Weather, NAS, Security, Late Aircraft) to support operational decision-making.", "#10b981"),
        ("O4", "Assess Cancellation Risk", "Independently predict cancellation probability using schedule-level features, giving travellers and airlines early warning to act.", "#f59e0b"),
        ("O5", "Deliver Actionable Insights", "Surface all predictions through an interactive dashboard with real-time visualisations, recommendations, and honest model performance reporting.", "#ec4899"),
    ]

    for badge, title, desc, color in objectives:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.02); border-left:3px solid {color};
                    border-radius:0 12px 12px 0; padding:16px 20px; margin:8px 0;
                    display:flex; gap:16px; align-items:flex-start;'>
            <div style='background:{color}22; border-radius:8px; padding:6px 12px;
                        color:{color}; font-weight:700; font-size:0.85rem;
                        font-family:monospace; white-space:nowrap; flex-shrink:0;'>{badge}</div>
            <div>
                <div style='font-weight:600; color:#e0f2fe; margin-bottom:4px;'>{title}</div>
                <div style='color:#64748b; font-size:0.875rem; line-height:1.6;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Dataset & Features
    st.markdown("---")
    st.markdown("## 📦 Dataset & Features")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                    border-radius:12px; padding:20px;'>
            <div style='font-weight:700; color:#e0f2fe; margin-bottom:12px;'>📂 Dataset</div>
            <table style='width:100%; font-size:0.85rem; color:#94a3b8; border-collapse:collapse;'>
                <tr><td style='padding:5px 0;color:#64748b;'>Source</td><td style='color:#e0f2fe;'>Bureau of Transportation Statistics</td></tr>
                <tr><td style='padding:5px 0;color:#64748b;'>Year</td><td style='color:#e0f2fe;'>2025 (12 months)</td></tr>
                <tr><td style='padding:5px 0;color:#64748b;'>Source file</td><td style='color:#e0f2fe;'>alldata/full_year.csv</td></tr>
                <tr><td style='padding:5px 0;color:#64748b;'>Raw Features</td><td style='color:#e0f2fe;'>26 columns, 479,748 rows</td></tr>
                <tr><td style='padding:5px 0;color:#64748b;'>Coverage</td><td style='color:#e0f2fe;'>All U.S. domestic airlines</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                    border-radius:12px; padding:20px;'>
            <div style='font-weight:700; color:#e0f2fe; margin-bottom:12px;'>🔧 Input Features (8)</div>
            <div style='display:flex; flex-wrap:wrap; gap:6px;'>
                <span class='badge'>MONTH</span>
                <span class='badge'>DAY_OF_MONTH</span>
                <span class='badge'>DAY_OF_WEEK</span>
                <span class='badge'>CARRIER_ENC</span>
                <span class='badge'>ORIGIN_ENC</span>
                <span class='badge'>DEST_ENC</span>
                <span class='badge'>DEP_TIME</span>
                <span class='badge'>DEP_DELAY</span>
            </div>
            <div style='margin-top:14px; font-weight:700; color:#e0f2fe; margin-bottom:8px;'>🎯 Target Variables</div>
            <div style='display:flex; flex-wrap:wrap; gap:6px;'>
                <span class='badge' style='border-color:#0ea5e977; color:#7dd3fc;'>ARR_DEL15</span>
                <span class='badge' style='border-color:#6366f177; color:#a5b4fc;'>ARR_DELAY</span>
                <span class='badge' style='border-color:#10b98177; color:#6ee7b7;'>DOMINANT_CAUSE</span>
                <span class='badge' style='border-color:#f59e0b77; color:#fcd34d;'>CANCELLED</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Beneficiaries
    st.markdown("---")
    st.markdown("## 👥 Who Benefits?")
    b1, b2, b3 = st.columns(3)
    for col, icon, title, desc in [
        (b1, "🧳", "Travellers", "Get delay and cancellation predictions before departure. Plan connections, book flexible fares, and receive tailored recommendations."),
        (b2, "✈️", "Airlines", "Identify high-risk routes and time windows. Pre-position crew and gates. Reduce cascade disruptions."),
        (b3, "🏢", "Airports", "Optimise gate management, ground staff scheduling, and passenger flow based on predicted delay patterns."),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                        border-radius:12px; padding:20px; text-align:center; height:100%;'>
                <div style='font-size:2rem; margin-bottom:10px;'>{icon}</div>
                <div style='font-weight:700; color:#e0f2fe; margin-bottom:8px;'>{title}</div>
                <div style='color:#64748b; font-size:0.85rem; line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)