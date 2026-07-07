import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from src.diagnostics import RootCauseDiagnostic   # <--- Properly Imported

st.set_page_config(
    page_title="Akino Connect — Telemetry Health Portal",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional SaaS Styling
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .saas-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .badge-ok {
        background-color: #ecfdf5;
        color: #059669;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #a7f3d0;
    }
    .badge-alert {
        background-color: #fef2f2;
        color: #dc2626;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #fecaca;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 0.8rem !important;
        font-weight: 600;
        text-transform: uppercase;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700;
        font-size: 1.9rem !important;
    }
    .chart-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px 8px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🏢 **Akino Connect**")
st.sidebar.caption("IT Infrastructure Management Services")
st.sidebar.markdown("---")

server_node = st.sidebar.selectbox("Active Blade / Host", [
    "prod-rack01-blade04.delhi.akino",
    "edge-gw02.blr.akino",
    "storage-san01.mumbai.akino"
])

sample_count = st.sidebar.slider("Sampling Window (Minutes)", 60, 480, 240, step=30)
has_spikes = st.sidebar.checkbox("Simulate Incident Spikes", value=True)
sensitivity = st.sidebar.slider("Anomaly Budget (%)", 1, 10, 4) / 100

# Telemetry Generator
def get_telemetry_data(n_samples=240, inject_fault=True):
    np.random.seed(42)
    timestamps = pd.date_range(end=datetime.now(), periods=n_samples, freq="min")
    
    cpu = 35 + 10 * np.sin(np.linspace(0, 12, n_samples)) + np.random.normal(0, 2.5, n_samples)
    temp = 52 + 4 * np.sin(np.linspace(0, 12, n_samples)) + np.random.normal(0, 1.5, n_samples)
    ram = 48 + 4 * np.linspace(0, 1.2, n_samples) + np.random.normal(0, 1.5, n_samples)
    latency = 18 + np.random.exponential(3.5, n_samples)
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "CPU_Load_%": np.clip(cpu, 5, 100).round(2),
        "Temperature_C": np.clip(temp, 35, 105).round(2),
        "RAM_Usage_%": np.clip(ram, 10, 100).round(2),
        "Network_Latency_ms": np.clip(latency, 5, 400).round(2)
    })
    
    if inject_fault:
        idx1 = int(n_samples * 0.35)
        df.loc[idx1:idx1+5, "Temperature_C"] = np.clip(df.loc[idx1:idx1+5, "Temperature_C"] + 38, 40, 96)
        df.loc[idx1:idx1+5, "CPU_Load_%"] = np.clip(df.loc[idx1:idx1+5, "CPU_Load_%"] + 42, 0, 98)
        
        idx2 = int(n_samples * 0.75)
        df.loc[idx2:idx2+6, "Network_Latency_ms"] += 175
        df.loc[idx2:idx2+6, "CPU_Load_%"] = np.clip(df.loc[idx2:idx2+6, "CPU_Load_%"] + 25, 0, 92)

    return df

raw_df = get_telemetry_data(sample_count, has_spikes)

# Unsupervised ML Engine
feature_cols = ["CPU_Load_%", "Temperature_C", "RAM_Usage_%", "Network_Latency_ms"]
iso_model = IsolationForest(n_estimators=100, contamination=sensitivity, random_state=42)
raw_df["Anomaly_Flag"] = iso_model.fit_predict(raw_df[feature_cols])
raw_df["Is_Incident"] = raw_df["Anomaly_Flag"] == -1
raw_df["Anomaly_Score"] = iso_model.decision_function(raw_df[feature_cols]).round(4)

# Execute Diagnostics Module Properly
df = RootCauseDiagnostic.annotate(raw_df)
incidents = df[df["Is_Incident"]].copy()
total_incidents = len(incidents)

# Top Bar
status_badge = (
    f'<span class="badge-alert">● {total_incidents} CRITICAL ANOMALIES DETECTED</span>'
    if total_incidents > 0
    else '<span class="badge-ok">● ALL SYSTEMS NOMINAL</span>'
)

st.markdown(f"""
<div class="saas-header">
    <div>
        <h2 style="margin:0; font-size: 1.45rem; font-weight:700; color:#0f172a;">
            Telemetry Observability & Failure Isolation
        </h2>
        <p style="margin: 4px 0 0 0; color:#64748b; font-size: 0.88rem;">
            Node: <code style="color:#0284c7; background:#f0f9ff; padding:2px 6px; border-radius:4px;">{server_node}</code>
            &nbsp;•&nbsp; Polling Interval: 60s &nbsp;•&nbsp; Engine: Isolation Forest
        </p>
    </div>
    <div>{status_badge}</div>
</div>
""", unsafe_allow_html=True)

# KPI Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("CPU Load", f"{df['CPU_Load_%'].iloc[-1]}%", delta=f"{df['CPU_Load_%'].max():.1f}% Peak")
c2.metric("Core Temp", f"{df['Temperature_C'].iloc[-1]} °C", delta=f"{df['Temperature_C'].max():.1f} °C Peak")
c3.metric("RAM Utilized", f"{df['RAM_Usage_%'].iloc[-1]}%", delta="Within SLA", delta_color="off")
c4.metric("Incident Window", f"{total_incidents} events", delta="Action Needed" if total_incidents > 0 else "Normal", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# Chart Creator
def create_chart(title, y_col, unit, stroke_color, fill_color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Timestamp"], y=df[y_col], mode="lines", name=title,
        line=dict(color=stroke_color, width=2), fill="tozeroy", fillcolor=fill_color
    ))
    if total_incidents > 0:
        fig.add_trace(go.Scatter(
            x=incidents["Timestamp"], y=incidents[y_col], mode="markers", name="Anomaly Flag",
            marker=dict(color="#ef4444", size=8, line=dict(color="#ffffff", width=1.5))
        ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b> <span style='font-size:12px; color:#94a3b8;'>({unit})</span>", font=dict(size=14, color="#1e293b")),
        template="plotly_white",
        height=240,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=10, color="#94a3b8")),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=10, color="#94a3b8")),
        showlegend=False,
        hovermode="x unified"
    )
    return fig

# 2x2 Grid
col_left, col_right = st.columns(2)
with col_left:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(create_chart("CPU Load Ratio", "CPU_Load_%", "%", "#2563eb", "rgba(37, 99, 235, 0.06)"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(create_chart("RAM Consumption", "RAM_Usage_%", "%", "#7c3aed", "rgba(124, 58, 237, 0.06)"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(create_chart("Core Thermal Reading", "Temperature_C", "°C", "#ea580c", "rgba(234, 88, 12, 0.06)"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(create_chart("Network Round-Trip Latency", "Network_Latency_ms", "ms", "#059669", "rgba(5, 150, 105, 0.06)"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Incident Audit Table
st.markdown("### 📋 Identified Incidents & Root Causes")
if total_incidents > 0:
    table_view = incidents[["Timestamp", "Root_Cause", "CPU_Load_%", "Temperature_C", "RAM_Usage_%", "Network_Latency_ms", "Anomaly_Score"]]
    st.dataframe(
        table_view.style.format({
            "CPU_Load_%": "{:.1f}%",
            "Temperature_C": "{:.1f}°C",
            "RAM_Usage_%": "{:.1f}%",
            "Network_Latency_ms": "{:.1f} ms",
            "Anomaly_Score": "{:.4f}"
        }),
        use_container_width=True
    )
else:
    st.success("Telemetry normal. Zero threshold deviations or multi-metric anomalies found.")