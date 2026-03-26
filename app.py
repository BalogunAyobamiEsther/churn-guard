import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ChurnGuard · 3MTT Osun",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background-color: #f7f8fa;
}

section[data-testid="stSidebar"] {
    background: #1a1d23 !important;
    border-right: 1px solid #2c3040;
    min-width: 210px !important;
    max-width: 210px !important;
}
section[data-testid="stSidebar"] > div { padding: 1.5rem 1.1rem 1rem; }

.sidebar-logo { margin-bottom: 1.8rem; }
.sidebar-logo .app-name { font-size: 0.95rem; font-weight: 600; color: #ffffff; }
.sidebar-logo .app-sub  { font-size: 0.68rem; color: #6b7280; margin-top: 2px; }

.sidebar-nav-label {
    font-size: 0.62rem; font-weight: 600; color: #FFFFFF;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem;
}


section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.84rem !important;
    font-weight: 400 !important;
    padding: 6px 10px !important;
    border-radius: 6px;
    cursor: pointer;
    display: block;
}

section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio label p,
section[data-testid="stSidebar"] .stRadio label span {
    color: #FFFFFF !important;
}           


section[data-testid="stSidebar"] .stRadio label:hover {
    background: #2c3040; color: #ffffff !important;
}

.sidebar-footer {
    margin-top: 2rem; padding-top: 0.9rem;
    border-top: 1px solid #2c3040;
    font-size: 0.68rem; color: #FFFFFF;
    line-height: 1.8; font-family: 'DM Mono', monospace;
}

.main .block-container {
    padding: 1.8rem 2.2rem 3rem;
    max-width: 980px;
    background: #f7f8fa;
}

.page-header {
    margin-bottom: 1.5rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid #e5e7eb;
}
.page-title { font-size: 1.45rem; font-weight: 600; color: #111827; letter-spacing: -0.02em; margin: 0 0 4px; }
.page-desc  { font-size: 0.85rem; color: #6b7280; margin: 0; }

.id-bar {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px;
    padding: 9px 15px; display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 1.4rem;
}
.id-bar .fn { font-size: 0.85rem; font-weight: 600; color: #111827; }
.id-bar .fm { font-size: 0.75rem; color: #6b7280; margin-top: 1px; }
.id-chip {
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 5px;
    font-family: 'DM Mono', monospace; font-size: 0.73rem; color: #1d4ed8;
    padding: 3px 9px; font-weight: 500;
}

.stat-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem 1.1rem; text-align: center; }
.stat-val   { font-size: 1.75rem; font-weight: 600; color: #111827; letter-spacing: -0.03em; line-height: 1; }
.stat-lbl   { font-size: 0.68rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 5px; }

.info-banner {
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
    padding: 11px 15px; font-size: 0.83rem; color: #1e40af;
    line-height: 1.6; margin-bottom: 1.2rem;
}

.sec-label {
    font-size: 0.67rem; font-weight: 600; color: #9ca3af;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.55rem;
}

.driver-wrap { display: flex; flex-wrap: wrap; gap: 5px; }
.driver-tag  {
    background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 5px;
    font-size: 0.76rem; color: #374151; padding: 4px 9px;
}

.fi-row {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px;
    padding: 0.8rem 1rem; margin-bottom: 0.55rem;
    display: flex; gap: 11px; align-items: flex-start;
}
.fi-row .fi-icon  { font-size: 1rem; margin-top: 2px; }
.fi-row .fi-title { font-size: 0.84rem; font-weight: 600; color: #111827; margin-bottom: 2px; }
.fi-row .fi-desc  { font-size: 0.78rem; color: #6b7280; line-height: 1.5; }

.result-high {
    background: #fffbeb; border: 1px solid #fcd34d;
    border-left: 4px solid #f59e0b; border-radius: 8px; padding: 1.1rem 1.3rem;
}
.result-low {
    background: #f0fdf4; border: 1px solid #86efac;
    border-left: 4px solid #22c55e; border-radius: 8px; padding: 1.1rem 1.3rem;
}
.result-verdict { font-size: 0.73rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 5px; }
.result-pct     { font-size: 2.4rem; font-weight: 700; letter-spacing: -0.04em; line-height: 1; }
.result-note    { font-size: 0.78rem; color: #6b7280; margin-top: 7px; }

.signal-item {
    font-size: 0.8rem; color: #374151; padding: 6px 10px;
    background: #ffffff; border: 1px solid #e5e7eb;
    border-left: 3px solid #f59e0b; border-radius: 0 6px 6px 0; margin-bottom: 4px;
}

.rec-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.85rem 0.95rem; margin-bottom: 0.55rem; }
.rec-title { font-size: 0.81rem; font-weight: 600; color: #111827; margin-bottom: 3px; }
.rec-body  { font-size: 0.77rem; color: #6b7280; line-height: 1.55; }

.detail-row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #f3f4f6; font-size: 0.82rem; }
.detail-row:last-child { border-bottom: none; }
.detail-key { color: #9ca3af; }
.detail-val { color: #111827; font-weight: 500; }

.tech-tag {
    display: inline-block; background: #f3f4f6; border: 1px solid #e5e7eb;
    border-radius: 4px; font-family: 'DM Mono', monospace;
    font-size: 0.7rem; color: #374151; padding: 3px 8px; margin: 2px 2px 2px 0;
}

.white-card { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:1rem 1.1rem; margin-bottom:0.8rem; }

.how-step { display: flex; gap: 13px; align-items: flex-start; padding: 0.85rem 0; border-bottom: 1px solid #f3f4f6; }
.how-step:last-child { border-bottom: none; }
.step-circle {
    min-width: 28px; height: 28px; border-radius: 50%;
    background: #eff6ff; border: 1px solid #bfdbfe;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.73rem; font-weight: 600; color: #1d4ed8; font-family: 'DM Mono', monospace;
}
.step-title { font-size: 0.84rem; font-weight: 600; color: #111827; margin-bottom: 2px; }
.step-body  { font-size: 0.78rem; color: #6b7280; line-height: 1.5; }

.tip-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.85rem 0.95rem; margin-bottom: 0.55rem; }
.tip-title { font-size: 0.81rem; font-weight: 600; color: #111827; margin-bottom: 3px; }
.tip-body  { font-size: 0.77rem; color: #6b7280; line-height: 1.5; }

div.stButton > button {
    background: #2563eb; color: #ffffff; border: none; border-radius: 7px;
    font-family: 'DM Sans', sans-serif; font-size: 0.875rem; font-weight: 500;
    padding: 0.6rem 1.5rem; width: 100%; transition: background 0.15s;
}
div.stButton > button:hover { background: #1d4ed8; }

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    font-size: 0.81rem !important; color: #374151 !important; font-weight: 500 !important;
}

hr { border-color: #e5e7eb !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="app-name">📡 ChurnGuard</div>
        <div class="app-sub">Telco Churn Predictor</div>
    </div>
    <div class="sidebar-nav-label">Menu</div>
    """, unsafe_allow_html=True)

    page = st.radio("nav", ["Home", "Predict Churn", "About", "How to Use"], label_visibility="collapsed")

    st.markdown("""
    <div class="sidebar-footer">
        Balogun Ayobami Esther<br>
        FE/23/58320157<br>
        Osun &nbsp;·&nbsp; 3MTT 2026
    </div>
    """, unsafe_allow_html=True)


def id_bar():
    st.markdown("""
    <div class="id-bar">
        <div>
            <div class="fn">Balogun Ayobami Esther</div>
            <div class="fm">Osun State &nbsp;·&nbsp; 3MTT NextGen Fellowship 2026 &nbsp;·&nbsp; Financial Inclusion</div>
        </div>
        <div class="id-chip">FE/23/58320157</div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    for path in ["data.csv", "/mnt/user-data/uploads/Telco_Customer_Churn_Dataset___3_.csv"]:
        try:
            df = pd.read_csv(path); break
        except Exception:
            df = None
    if df is None:
        return None, None, None
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(object)
    df.dropna(inplace=True)
    df = df.drop("customerID", axis=1)
    enc = df.copy()
    le = LabelEncoder()
    for col in enc.select_dtypes(include="object").columns:
        enc[col] = le.fit_transform(enc[col])
    X = enc.drop("Churn", axis=1)
    y = enc["Churn"]
    cols = X.columns.tolist()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])
    model.fit(Xtr, ytr)
    yp = model.predict(Xte); yprob = model.predict_proba(Xte)[:, 1]
    rpt = classification_report(yte, yp, output_dict=True)
    m = {"acc": round(accuracy_score(yte, yp)*100,1), "auc": round(roc_auc_score(yte, yprob)*100,1),
         "prec": round(rpt["1"]["precision"]*100,1), "rec": round(rpt["1"]["recall"]*100,1)}
    return model, cols, m


ENCODE = {
    "gender": {"Female":0,"Male":1}, "SeniorCitizen": {"No":0,"Yes":1},
    "Partner": {"No":0,"Yes":1}, "Dependents": {"No":0,"Yes":1},
    "PhoneService": {"No":0,"Yes":1},
    "MultipleLines": {"No":0,"No phone service":1,"Yes":2},
    "InternetService": {"DSL":0,"Fiber optic":1,"No":2},
    "OnlineSecurity": {"No":0,"No internet service":1,"Yes":2},
    "OnlineBackup": {"No":0,"No internet service":1,"Yes":2},
    "DeviceProtection": {"No":0,"No internet service":1,"Yes":2},
    "TechSupport": {"No":0,"No internet service":1,"Yes":2},
    "StreamingTV": {"No":0,"No internet service":1,"Yes":2},
    "StreamingMovies": {"No":0,"No internet service":1,"Yes":2},
    "Contract": {"Month-to-month":0,"One year":1,"Two year":2},
    "PaperlessBilling": {"No":0,"Yes":1},
    "PaymentMethod": {"Bank transfer (automatic)":0,"Credit card (automatic)":1,"Electronic check":2,"Mailed check":3},
}

def prep(ui, cols):
    row = {c: ENCODE[c].get(ui.get(c,0), 0) if c in ENCODE else ui.get(c,0) for c in cols}
    return pd.DataFrame([row])


# HOME
if page == "Home":
    id_bar()
    st.markdown('<div class="page-header"><p class="page-title">Customer Churn Intelligence</p><p class="page-desc">Predicting telco churn to protect mobile-based financial access in Nigeria.</p></div>', unsafe_allow_html=True)

    with st.spinner(""):
        model, cols, m = load_model()

    if m:
        st.markdown('<div class="sec-label">Model Performance</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in zip([c1,c2,c3,c4],[m["acc"],m["auc"],m["prec"],m["rec"]],["Accuracy","ROC-AUC","Precision","Recall"]):
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-val">{val}%</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        st.markdown('<div class="sec-label">Dataset</div>', unsafe_allow_html=True)
        st.markdown('<div class="white-card"><div class="detail-row"><span class="detail-key">Records</span><span class="detail-val">7,032</span></div><div class="detail-row"><span class="detail-key">Features</span><span class="detail-val">20</span></div><div class="detail-row"><span class="detail-key">Churn rate</span><span class="detail-val">26.6%</span></div><div class="detail-row"><span class="detail-key">Model</span><span class="detail-val">Logistic Regression</span></div></div>', unsafe_allow_html=True)
    with r:
        st.markdown('<div class="sec-label">Top Churn Drivers</div>', unsafe_allow_html=True)
        st.markdown('<div class="white-card"><div class="driver-wrap"><span class="driver-tag">Month-to-month contract</span><span class="driver-tag">High monthly charges</span><span class="driver-tag">Short tenure</span><span class="driver-tag">Fiber optic</span><span class="driver-tag">Electronic check</span><span class="driver-tag">No online security</span></div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Financial Inclusion Link</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fi-row"><div class="fi-icon">📱</div><div><div class="fi-title">Mobile money access</div><div class="fi-desc">Churned customers lose access to Airtel Money, MTN MoMo, and USSD banking.</div></div></div>
    <div class="fi-row"><div class="fi-icon">🏘</div><div><div class="fi-title">Rural communities</div><div class="fi-desc">In underserved areas, a mobile line is the only gateway to financial services.</div></div></div>
    <div class="fi-row"><div class="fi-icon">🎯</div><div><div class="fi-title">Targeted retention</div><div class="fi-desc">This model flags at-risk customers so telcos can intervene before churn happens.</div></div></div>
    """, unsafe_allow_html=True)


# PREDICT
elif page == "Predict Churn":
    id_bar()
    st.markdown('<div class="page-header"><p class="page-title">Predict Churn Risk</p><p class="page-desc">Score a single customer or upload a CSV for bulk predictions.</p></div>', unsafe_allow_html=True)

    with st.spinner(""):
        model, cols, m = load_model()
    if model is None:
        st.error("Dataset not found. Place data.csv in the same folder as app.py.")
        st.stop()

    # ── Toggle 
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"]:has(.toggle-btn) { gap: 0 !important; }
    .toggle-btn button {
        border-radius: 0 !important; border: 1px solid #e5e7eb !important;
        background: #ffffff !important; color: #6b7280 !important;
        font-size: 0.82rem !important; font-weight: 500 !important;
        padding: 0.45rem 1.2rem !important; width: 100% !important;
        transition: all 0.15s !important;
    }
    .toggle-btn button:hover { background: #f3f4f6 !important; color: #111827 !important; }
    .toggle-active button {
        background: #2563eb !important; color: #ffffff !important;
        border-color: #2563eb !important;
    }
    .toggle-left  button { border-radius: 7px 0 0 7px !important; }
    .toggle-right button { border-radius: 0 7px 7px 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    if "predict_mode" not in st.session_state:
        st.session_state.predict_mode = "bulk"

    tb_r, tb_l, tb_spacer = st.columns([1, 1, 5])

    with tb_r:
        active_r = "toggle-btn toggle-right toggle-active" if st.session_state.predict_mode == "bulk" else "toggle-btn toggle-right"
        st.markdown(f'<div class="{active_r}">', unsafe_allow_html=True)
        if st.button("Bulk Upload"):
            st.session_state.predict_mode = "bulk"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tb_l:
        active_l = "toggle-btn toggle-left toggle-active" if st.session_state.predict_mode == "single" else "toggle-btn toggle-left"
        st.markdown(f'<div class="{active_l}">', unsafe_allow_html=True)
        if st.button("Single Customer"):
            st.session_state.predict_mode = "single"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    

    st.markdown("<br>", unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════════
    # BULK PREDICTION
    # ════════════════════════════════════════════════════════════════════════
    if st.session_state.predict_mode == "bulk":
        # Expected columns info
        st.markdown("""
        <div class="info-banner" style="margin-bottom:1rem;">
            Upload a CSV with the same columns as the training dataset. A template is available to download below.
            The <strong>customerID</strong> column is optional — if present it will be kept as an identifier.
        </div>
        """, unsafe_allow_html=True)

        # Template download
        template_cols = [
            "customerID","gender","SeniorCitizen","Partner","Dependents","tenure",
            "PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
            "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
            "Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"
        ]
        sample_rows = [
            ["C001","Female","0","Yes","No","12","Yes","No","DSL","No","Yes","No","No","No","No","Month-to-month","Yes","Electronic check","65.0","780.0"],
            ["C002","Male","0","No","No","48","Yes","Yes","Fiber optic","Yes","No","Yes","Yes","Yes","Yes","Two year","No","Credit card (automatic)","89.5","4296.0"],
            ["C003","Female","1","Yes","Yes","2","Yes","No","DSL","No","No","No","No","No","No","Month-to-month","Yes","Mailed check","45.0","90.0"],
        ]
        template_df = pd.DataFrame(sample_rows, columns=template_cols)
        template_csv = template_df.to_csv(index=False)

        dl_col, _ = st.columns([1, 3])
        with dl_col:
            st.download_button(
                label="Download CSV Template",
                data=template_csv,
                file_name="churnguard_template.csv",
                mime="text/csv",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"], label_visibility="collapsed")

        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
                has_id = "customerID" in raw.columns
                id_col = raw["customerID"].astype(str) if has_id else pd.Series([f"Row {i+1}" for i in range(len(raw))], name="customerID")

                # Pre-process
                df_bulk = raw.copy()
                if "customerID" in df_bulk.columns:
                    df_bulk = df_bulk.drop("customerID", axis=1)
                if "Churn" in df_bulk.columns:
                    df_bulk = df_bulk.drop("Churn", axis=1)

                df_bulk["TotalCharges"] = pd.to_numeric(df_bulk["TotalCharges"], errors="coerce")
                df_bulk["SeniorCitizen"] = df_bulk["SeniorCitizen"].astype(str).str.strip()

                # Encode
                df_enc = df_bulk.copy()
                for c in df_enc.columns:
                    if c in ENCODE:
                        df_enc[c] = df_enc[c].map(ENCODE[c]).fillna(0).astype(int)

                # Align columns
                for c in cols:
                    if c not in df_enc.columns:
                        df_enc[c] = 0
                df_enc = df_enc[cols].fillna(0)

                probs = model.predict_proba(df_enc)[:, 1]
                preds = model.predict(df_enc)

                # Build results
                results = pd.DataFrame({
                    "Customer ID": id_col.values,
                    "Churn Risk %": (probs * 100).round(1),
                    "Prediction": ["High Risk" if p == 1 else "Low Risk" for p in preds],
                })

                n_total  = len(results)
                n_high   = int((preds == 1).sum())
                n_low    = n_total - n_high
                avg_risk = round(float(probs.mean() * 100), 1)

                # Summary stats
                st.markdown("---")
                st.markdown("**Batch Summary**")
                s1,s2,s3,s4 = st.columns(4)
                s1.markdown(f'<div class="stat-card"><div class="stat-val">{n_total}</div><div class="stat-lbl">Total Customers</div></div>', unsafe_allow_html=True)
                s2.markdown(f'<div class="stat-card"><div class="stat-val" style="color:#b45309;">{n_high}</div><div class="stat-lbl">High Risk</div></div>', unsafe_allow_html=True)
                s3.markdown(f'<div class="stat-card"><div class="stat-val" style="color:#15803d;">{n_low}</div><div class="stat-lbl">Low Risk</div></div>', unsafe_allow_html=True)
                s4.markdown(f'<div class="stat-card"><div class="stat-val">{avg_risk}%</div><div class="stat-lbl">Avg Churn Risk</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Colour the table
                def colour_row(row):
                    if row["Prediction"] == "High Risk":
                        return ["", "color: #b45309; font-weight:600", "color: #b45309; font-weight:600"]
                    return ["", "color: #15803d; font-weight:600", "color: #15803d; font-weight:600"]

                st.markdown("**Results Table**")
                st.dataframe(
                    results.sort_values("Churn Risk %", ascending=False).reset_index(drop=True),
                    use_container_width=True,
                    height=min(400, 40 + n_total * 36),
                )

                # Download results
                st.markdown("<br>", unsafe_allow_html=True)
                dl2, _ = st.columns([1, 3])
                with dl2:
                    st.download_button(
                        label="Download Results CSV",
                        data=results.sort_values("Churn Risk %", ascending=False).to_csv(index=False),
                        file_name="churnguard_results.csv",
                        mime="text/csv",
                    )

                # High-risk list
                high_risk_df = results[results["Prediction"] == "High Risk"].sort_values("Churn Risk %", ascending=False)
                if not high_risk_df.empty:
                    st.markdown("---")
                    st.markdown(f"**Top High-Risk Customers** ({len(high_risk_df)} flagged)")
                    for _, row in high_risk_df.head(10).iterrows():
                        st.markdown(f"""
                        <div style="display:flex;justify-content:space-between;align-items:center;
                                    padding:8px 12px;background:#fffbeb;border:1px solid #fcd34d;
                                    border-left:3px solid #f59e0b;border-radius:0 6px 6px 0;margin-bottom:4px;">
                            <span style="font-size:0.82rem;color:#374151;font-weight:500;">{row['Customer ID']}</span>
                            <span style="font-size:0.82rem;color:#b45309;font-weight:600;">{row['Churn Risk %']}% risk</span>
                        </div>""", unsafe_allow_html=True)
                    if len(high_risk_df) > 10:
                        st.markdown(f'<div style="font-size:0.78rem;color:#9ca3af;margin-top:4px;">+ {len(high_risk_df)-10} more in the downloaded CSV.</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Could not process file: {e}. Please check the CSV format and try again.")



    # ════════════════════════════════════════════════════════════════════════
    # SINGLE PREDICTION
    # ════════════════════════════════════════════════════════════════════════
    else:
        with st.form("churn_form"):
            st.markdown("**Demographics**")
            a1,a2,a3,a4,a5 = st.columns(5)
            gender   = a1.selectbox("Gender",          ["Female","Male"])
            senior   = a2.selectbox("Senior Citizen",  ["No","Yes"])
            partner  = a3.selectbox("Partner",          ["No","Yes"])
            deps     = a4.selectbox("Dependents",       ["No","Yes"])
            tenure   = a5.slider("Tenure (months)", 0, 72, 12)

            st.markdown("---")
            st.markdown("**Services**")
            b1,b2,b3 = st.columns(3)
            phone    = b1.selectbox("Phone Service",    ["No","Yes"])
            lines    = b2.selectbox("Multiple Lines",   ["No","Yes","No phone service"])
            internet = b3.selectbox("Internet Service", ["DSL","Fiber optic","No"])

            c1,c2,c3,c4 = st.columns(4)
            security = c1.selectbox("Online Security",    ["No","Yes","No internet service"])
            backup   = c2.selectbox("Online Backup",      ["No","Yes","No internet service"])
            device   = c3.selectbox("Device Protection",  ["No","Yes","No internet service"])
            tech     = c4.selectbox("Tech Support",       ["No","Yes","No internet service"])

            d1,d2 = st.columns(2)
            tv     = d1.selectbox("Streaming TV",      ["No","Yes","No internet service"])
            movies = d2.selectbox("Streaming Movies",  ["No","Yes","No internet service"])

            st.markdown("---")
            st.markdown("**Billing & Contract**")
            e1,e2,e3 = st.columns(3)
            contract  = e1.selectbox("Contract",          ["Month-to-month","One year","Two year"])
            paperless = e2.selectbox("Paperless Billing", ["No","Yes"])
            payment   = e3.selectbox("Payment Method",    ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])

            f1,f2 = st.columns(2)
            monthly = f1.number_input("Monthly Charges (₦)", 18.0, 120.0, 65.0, 0.5)
            total   = f2.number_input("Total Charges (₦)",    0.0, 9000.0, float(monthly*tenure), 10.0)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Generate Risk Report")

        if submitted:
            ui = dict(gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=deps,
                      tenure=tenure, PhoneService=phone, MultipleLines=lines, InternetService=internet,
                      OnlineSecurity=security, OnlineBackup=backup, DeviceProtection=device, TechSupport=tech,
                      StreamingTV=tv, StreamingMovies=movies, Contract=contract, PaperlessBilling=paperless,
                      PaymentMethod=payment, MonthlyCharges=monthly, TotalCharges=total)
            Xi   = prep(ui, cols)
            prob = model.predict_proba(Xi)[0][1]
            pred = int(model.predict(Xi)[0])
            pct  = round(prob*100, 1)

            st.markdown("---")
            st.markdown("**Result**")
            r1,r2 = st.columns(2)
            with r1:
                if pred == 1:
                    st.markdown(f'<div class="result-high"><div class="result-verdict" style="color:#b45309;">High Churn Risk</div><div class="result-pct" style="color:#92400e;">{pct}%</div><div class="result-note">Retention action recommended.</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-low"><div class="result-verdict" style="color:#15803d;">Low Churn Risk</div><div class="result-pct" style="color:#166534;">{pct}%</div><div class="result-note">Customer appears stable.</div></div>', unsafe_allow_html=True)

            with r2:
                signals = []
                if contract == "Month-to-month":   signals.append("Month-to-month contract")
                if tenure < 12:                    signals.append(f"Short tenure — {tenure} months")
                if monthly > 70:                   signals.append(f"High monthly charges — ₦{monthly}")
                if internet == "Fiber optic":      signals.append("Fiber optic internet user")
                if payment == "Electronic check":  signals.append("Electronic check payment")
                if security == "No":               signals.append("No online security")
                if tech == "No":                   signals.append("No tech support")
                if signals:
                    st.markdown('<div class="sec-label">Risk Signals</div>', unsafe_allow_html=True)
                    for s in signals:
                        st.markdown(f'<div class="signal-item">{s}</div>', unsafe_allow_html=True)

            if pred == 1:
                st.markdown("---")
                st.markdown("**Retention Recommendations**")
                recs = []
                if contract == "Month-to-month":     recs.append(("Offer contract upgrade",  "Provide a discount to switch to a 12 or 24-month plan."))
                if monthly > 70:                     recs.append(("Bundle repricing",        "Offer a service bundle to reduce effective monthly cost."))
                if tenure < 12:                      recs.append(("Early engagement",        "Enrol in a 90-day onboarding programme with welcome incentives."))
                if security == "No" or tech == "No": recs.append(("Free add-on trial",      "Offer a 30-day free trial of Online Security or Tech Support."))
                if payment == "Electronic check":    recs.append(("Auto-pay incentive",      "Offer a small monthly credit to switch to automatic payment."))
                if not recs:                         recs.append(("Proactive check-in",      "Conduct a satisfaction survey — this profile warrants monitoring."))
                rc1,rc2 = st.columns(2)
                for i,(t,b) in enumerate(recs):
                    with rc1 if i%2==0 else rc2:
                        st.markdown(f'<div class="rec-card"><div class="rec-title">{t}</div><div class="rec-body">{b}</div></div>', unsafe_allow_html=True)


# ABOUT
elif page == "About":
    id_bar()
    st.markdown('<div class="page-header"><p class="page-title">About ChurnGuard</p><p class="page-desc">3MTT NextGen Knowledge Showcase &nbsp;·&nbsp; Financial Inclusion Pillar &nbsp;·&nbsp; Osun State</p></div>', unsafe_allow_html=True)

    l,r = st.columns([3,2])
    with l:
        st.markdown('<div class="sec-label">Problem</div>', unsafe_allow_html=True)
        st.markdown('<div class="white-card"><p style="font-size:0.84rem;color:#374151;line-height:1.75;margin:0;">Telco churn is a financial inclusion problem. Millions of Nigerians use their mobile lines as their only gateway to USSD banking, mobile money, and digital payments. When they churn, they lose that access.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Solution</div>', unsafe_allow_html=True)
        st.markdown('<div class="white-card"><p style="font-size:0.84rem;color:#374151;line-height:1.75;margin:0;">ChurnGuard predicts which customers are at risk of leaving using a Logistic Regression model trained on 7,032 telco records. It surfaces specific risk signals and generates personalised retention recommendations framed around financial inclusion outcomes.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Technical Approach</div>', unsafe_allow_html=True)
        st.markdown('<div class="white-card"><p style="font-size:0.84rem;color:#374151;line-height:1.75;margin:0;">Five models were evaluated: Logistic Regression, Decision Tree, Random Forest, KNN, and XGBoost — each with class balancing and GridSearchCV tuning. The final Pipeline (StandardScaler + Logistic Regression) achieves 83.4% ROC-AUC and 79.4% recall on the churn class.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Tech Stack</div>', unsafe_allow_html=True)
        st.markdown('<div class="white-card"><span class="tech-tag">Python 3</span><span class="tech-tag">Streamlit</span><span class="tech-tag">scikit-learn</span><span class="tech-tag">pandas</span><span class="tech-tag">numpy</span><span class="tech-tag">XGBoost</span><span class="tech-tag">GridSearchCV</span><span class="tech-tag">Pipeline</span></div>', unsafe_allow_html=True)

    with r:
        st.markdown('<div class="sec-label">Fellow Details</div>', unsafe_allow_html=True)
        st.markdown('<div class="white-card"><div style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:2px;">Bankole Ridwan</div><div style="font-family:\'DM Mono\',monospace;font-size:0.76rem;color:#2563eb;margin-bottom:1rem;">osun/2026/0009</div><div class="detail-row"><span class="detail-key">State</span><span class="detail-val">Osun</span></div><div class="detail-row"><span class="detail-key">Programme</span><span class="detail-val">3MTT NextGen 2026</span></div><div class="detail-row"><span class="detail-key">Pillar</span><span class="detail-val">Financial Inclusion</span></div><div class="detail-row"><span class="detail-key">Partner</span><span class="detail-val">Airtel Africa · NITDA</span></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Dataset</div>', unsafe_allow_html=True)
        st.markdown('<div class="white-card"><div class="detail-row"><span class="detail-key">Records</span><span class="detail-val">7,032</span></div><div class="detail-row"><span class="detail-key">Features</span><span class="detail-val">20</span></div><div class="detail-row"><span class="detail-key">Target</span><span class="detail-val">Churn (Yes/No)</span></div><div class="detail-row"><span class="detail-key">Churn rate</span><span class="detail-val">26.6%</span></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Impact</div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:0.9rem 1rem;"><p style="font-size:0.81rem;color:#166534;line-height:1.6;margin:0;">This model can be containerised and integrated into any telco CRM for real-time churn scoring — enabling retention teams to act before financially vulnerable customers lose their mobile financial access.</p></div>', unsafe_allow_html=True)


# HOW TO USE
elif page == "How to Use":
    id_bar()
    st.markdown('<div class="page-header"><p class="page-title">How to Use ChurnGuard</p><p class="page-desc">A quick guide to generating a churn risk prediction.</p></div>', unsafe_allow_html=True)

    l,r = st.columns([3,2])
    with l:
        st.markdown('<div class="sec-label">Steps</div>', unsafe_allow_html=True)
        steps = [
            ("Go to Predict Churn",        "Select it from the sidebar navigation on the left."),
            ("Enter demographics",          "Fill in gender, senior status, partner, dependents, and tenure."),
            ("Select services",             "Choose phone, internet type, and any add-on services."),
            ("Enter billing details",       "Select contract type, payment method, monthly and total charges."),
            ("Click Generate Risk Report",  "The model scores the customer instantly."),
            ("Read the risk score",         "Amber = high risk. Green = low risk."),
            ("Review risk signals",         "Factors driving the score are listed on the right side."),
            ("Act on recommendations",      "If high risk, personalised retention actions appear below."),
        ]
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        for i,(t,b) in enumerate(steps, 1):
            st.markdown(f'<div class="how-step"><div class="step-circle">{i:02d}</div><div><div class="step-title">{t}</div><div class="step-body">{b}</div></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r:
        st.markdown('<div class="sec-label">Reading the Output</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#fffbeb;border:1px solid #fcd34d;border-left:4px solid #f59e0b;border-radius:8px;padding:0.85rem 1rem;margin-bottom:0.6rem;">
            <div style="font-size:0.72rem;font-weight:600;color:#b45309;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:3px;">High Risk (Amber)</div>
            <div style="font-size:0.79rem;color:#78350f;">Score above ~50%. Act on the recommendations shown below the result.</div>
        </div>
        <div style="background:#f0fdf4;border:1px solid #86efac;border-left:4px solid #22c55e;border-radius:8px;padding:0.85rem 1rem;margin-bottom:1rem;">
            <div style="font-size:0.72rem;font-weight:600;color:#15803d;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:3px;">Low Risk (Green)</div>
            <div style="font-size:0.79rem;color:#166534;">Score below ~50%. Customer is stable. Monitor if charges or contract change.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-label">Tips</div>', unsafe_allow_html=True)
        for t,b in [
            ("Contract type matters most",   "Month-to-month is the strongest single churn predictor."),
            ("Check early-tenure customers", "Customers under 12 months are most at risk."),
            ("Use realistic charge values",  "Monthly charges heavily influence the prediction score."),
        ]:
            st.markdown(f'<div class="tip-card"><div class="tip-title">{t}</div><div class="tip-body">{b}</div></div>', unsafe_allow_html=True) 