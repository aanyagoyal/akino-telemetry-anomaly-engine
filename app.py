import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from src.diagnostics import RootCauseDiagnostic

st.set_page_config(
    page_title="Akino Connect — Infrastructure Telemetry Monitor",
    layout="wide"
)

# Sidebar
st.sidebar.title("Akino Connect")
st.sidebar.caption("IT Infrastructure Monitoring Portal")
st.sidebar.markdown("---")

server_node = st.sidebar.selectbox("Monitored Node", [
    "rack01-blade04.internal",
    "edge-gw02.internal",
    "storage-san01.internal"
])

sample_count = st.sidebar.slider("Monitor Window (Minutes)", 60, 480, 240, step=30)
has_spikes = st.sidebar.checkbox("Simulate Incident Spikes", value=True)
sensitivity = st.sidebar.slider("Anomaly Sensitivity (%)", 1, 10, 4) / 100

# Telemetry Generator with Packet Loss
def get_telemetry_data(n_samples=240, inject_fault=True):
    np.random.seed(42)
    timestamps = pd.date_range(end=datetime.now(), periods=n_samples, freq="min")
    
    cpu = 35 + 10 * np.sin(np.linspace(0, 12, n_samples)) + np.random.normal(0, 2.5, n_samples)
    temp = 52 + 4 * np.sin(np.linspace(0, 12, n_samples)) + np.random.normal(0, 1.5, n_samples)
    ram = 48 + 4 * np.linspace(0, 1.2, n_samples) + np.random.normal(0, 1.5, n_samples)
    latency = 18 + np.random.exponential(3.5, n_samples)
    packet_loss = np.random.exponential(0.3, n_samples)  # Base packet loss ~0.3%
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "CPU_Load_%": np.clip(cpu, 5, 100).round(2),
        "Temperature_C": np.clip(temp, 35, 105).round(2),
        "RAM_Usage_%": np.clip(ram, 10, 100).round(2),
        "Network_Latency_ms": np.clip(latency, 5, 400).round(2),
        "Packet_Loss_%": np.clip(packet_loss, 0, 100).round(2)
    })
    
    if inject_fault:
        # Incident 1: Overheating & Compute Load
        idx1 = int(n_samples * 0.35)
        df.loc[idx1:idx1+5, "Temperature_C"] = np.clip(df.loc[idx1:idx1+5, "Temperature_C"] + 38, 40, 96)
        df.loc[idx1:idx1+5, "CPU_Load_%"] = np.clip(df.loc[idx1:idx1+5, "CPU_Load_%"] + 42, 0, 98)
        
        # Incident 2: Network Congestion & High Packet Loss
        idx2 = int(n_samples * 0.75)
        df.loc[idx2:idx2+6, "Network_Latency_ms"] += 175
        df.loc[idx2:idx2+6, "Packet_Loss_%"] = np.clip(df.loc[idx2:idx2+6, "Packet_Loss_%"] + 12.5, 0, 85)
        df.loc[idx2:idx2+6, "CPU_Load_%"] = np.clip(df.loc[idx2:idx2+6, "CPU_Load_%"] + 25, 0, 92)

    return df

# Data Pipeline
raw_df = get_telemetry_data(sample_count, has_spikes)

# Model Training (Including Packet_Loss_%)
feature_cols = ["CPU_Load_%", "Temperature_C", "RAM_Usage_%", "Network_Latency_ms", "Packet_Loss_%"]
iso_model = IsolationForest(n_estimators=100, contamination=sensitivity, random_state=42)
raw_df["Anomaly_Flag"] = iso_model.fit_predict(raw_df[feature_cols])
raw_df["Is_Incident"] = raw_df["Anomaly_Flag"] == -1
raw_df["Anomaly_Score"] = iso_model.decision_function(raw_df[feature_cols]).round(4)

# Diagnostics Engine
df = RootCauseDiagnostic.annotate(raw_df)
incidents = df[df["Is_Incident"]].copy()
total_incidents = len(incidents)

# Header
st.title("Server Health & Telemetry Anomaly Monitor")
st.caption(f"Cluster Node: {server_node} | Engine: Unsupervised Isolation Forest (5 Feature Vectors)")

# KPI Summary Row (5 Metrics)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("CPU Load", f"{df['CPU_Load_%'].iloc[-1]}%")
k2.metric("Core Temp", f"{df['Temperature_C'].iloc[-1]} °C")
k3.metric("RAM Utilized", f"{df['RAM_Usage_%'].iloc[-1]}%")
k4.metric("Packet Loss", f"{df['Packet_Loss_%'].iloc[-1]}%", delta=f"{df['Packet_Loss_%'].max():.1f}% Peak", delta_color="inverse")
k5.metric("Flagged Incidents", f"{total_incidents} events", delta="Action Required" if total_incidents > 0 else "Optimal", delta_color="inverse")

st.markdown("---")

# Chart Maker Function
def render_metric_plot(title, col_name, unit, line_color):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["Timestamp"], 
        y=df[col_name], 
        mode="lines", 
        name=title,
        line=dict(color=line_color, width=2)
    ))
    
    if total_incidents > 0:
        fig.add_trace(go.Scatter(
            x=incidents["Timestamp"], 
            y=incidents[col_name], 
            mode="markers", 
            name="Anomaly",
            marker=dict(color="#d63031", size=8, symbol="circle")
        ))
        
    fig.update_layout(
        title=f"<b>{title}</b> ({unit})",
        height=240,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_title="",
        yaxis_title=unit,
        showlegend=False,
        hovermode="x unified"
    )
    return fig

# Charts Display (Row 1: Hardware Metrics)
r1_c1, r1_c2, r1_c3 = st.columns(3)
with r1_c1:
    st.plotly_chart(render_metric_plot("CPU Load Ratio", "CPU_Load_%", "%", "#0984e3"), use_container_width=True)
with r1_c2:
    st.plotly_chart(render_metric_plot("Core Thermal Reading", "Temperature_C", "°C", "#e17055"), use_container_width=True)
with r1_c3:
    st.plotly_chart(render_metric_plot("RAM Consumption", "RAM_Usage_%", "%", "#6c5ce7"), use_container_width=True)

# Charts Display (Row 2: Network Metrics)
r2_c1, r2_c2 = st.columns(2)
with r2_c1:
    st.plotly_chart(render_metric_plot("Network Latency", "Network_Latency_ms", "ms", "#00b894"), use_container_width=True)
with r2_c2:
    st.plotly_chart(render_metric_plot("Network Packet Loss", "Packet_Loss_%", "%", "#d63031"), use_container_width=True)

# Incident Table
st.markdown("---")
st.subheader("Identified Incident Log & Root Cause Analysis")

if total_incidents > 0:
    table_view = incidents[["Timestamp", "Root_Cause", "CPU_Load_%", "Temperature_C", "RAM_Usage_%", "Network_Latency_ms", "Packet_Loss_%"]]
    st.dataframe(table_view, use_container_width=True)
else:
    st.success("All systems nominal. No correlated multi-metric anomalies found.")