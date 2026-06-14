import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go

st.set_page_config(page_title="Akino Connect - Server Health Monitor", layout="wide")

# Sidebar
st.sidebar.title("Akino Connect")
st.sidebar.caption("IT Infrastructure Monitoring Portal")
st.sidebar.markdown("---")

sample_count = st.sidebar.slider("Monitor Window (Minutes)", 60, 480, 240, step=30)
has_spikes = st.sidebar.checkbox("Simulate Server Outage / Spikes", value=True)
sensitivity = st.sidebar.slider("Anomaly Sensitivity (%)", 1, 10, 4) / 100

# Direct Data Generator (Zero Import Errors)
def get_telemetry_data(n_samples=240, inject_fault=True):
    np.random.seed(42)
    timestamps = pd.date_range(end=datetime.now(), periods=n_samples, freq="min")
    
    cpu = 35 + 10 * np.sin(np.linspace(0, 12, n_samples)) + np.random.normal(0, 3, n_samples)
    temp = 55 + 5 * np.sin(np.linspace(0, 12, n_samples)) + np.random.normal(0, 2, n_samples)
    ram = 50 + 5 * np.linspace(0, 1.2, n_samples) + np.random.normal(0, 2, n_samples)
    latency = 20 + np.random.exponential(4, n_samples)
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "CPU_Load_%": np.clip(cpu, 5, 100).round(2),
        "Temperature_C": np.clip(temp, 35, 105).round(2),
        "RAM_Usage_%": np.clip(ram, 10, 100).round(2),
        "Network_Latency_ms": np.clip(latency, 5, 400).round(2)
    })
    
    if inject_fault:
        idx1 = int(n_samples * 0.35)
        df.loc[idx1:idx1+5, "Temperature_C"] = np.clip(df.loc[idx1:idx1+5, "Temperature_C"] + 35, 40, 96)
        df.loc[idx1:idx1+5, "CPU_Load_%"] = np.clip(df.loc[idx1:idx1+5, "CPU_Load_%"] + 45, 0, 98)
        
        idx2 = int(n_samples * 0.75)
        df.loc[idx2:idx2+6, "Network_Latency_ms"] += 180
        df.loc[idx2:idx2+6, "CPU_Load_%"] = np.clip(df.loc[idx2:idx2+6, "CPU_Load_%"] + 30, 0, 95)

    return df

# Generate Data
df = get_telemetry_data(sample_count, has_spikes)

# Model Training
feature_cols = ["CPU_Load_%", "Temperature_C", "RAM_Usage_%", "Network_Latency_ms"]
iso_model = IsolationForest(n_estimators=100, contamination=sensitivity, random_state=42)
df["Anomaly_Flag"] = iso_model.fit_predict(df[feature_cols])
df["Is_Incident"] = df["Anomaly_Flag"] == -1
incidents = df[df["Is_Incident"]]

# Header
st.title("🖥️ Server Health & Telemetry Anomaly Monitor")
st.write("Real-time telemetry stream from **Rack-Server-01**. Spikes identified by the Isolation Forest model are highlighted in red.")

# Summary Metrics Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current CPU Load", f"{df['CPU_Load_%'].iloc[-1]} %")
c2.metric("Server Temperature", f"{df['Temperature_C'].iloc[-1]} °C")
c3.metric("RAM Usage", f"{df['RAM_Usage_%'].iloc[-1]} %")
c4.metric("Flagged Outliers", int(df["Is_Incident"].sum()), delta="Incidents Found", delta_color="inverse")

st.markdown("---")

# Chart Maker Function
def create_metric_chart(title, y_col, unit, line_color):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["Timestamp"], 
        y=df[y_col], 
        mode="lines", 
        name=f"Normal {title}",
        line=dict(color=line_color, width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=incidents["Timestamp"], 
        y=incidents[y_col], 
        mode="markers", 
        name="Anomaly Detected",
        marker=dict(color="red", size=8, symbol="circle")
    ))
    
    fig.update_layout(
        title=f"<b>{title} ({unit})</b>",
        xaxis_title="Time",
        yaxis_title=unit,
        template="plotly_white",
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# 2x2 Clean Grid
r1_c1, r1_c2 = st.columns(2)
with r1_c1:
    st.plotly_chart(create_metric_chart("CPU Load", "CPU_Load_%", "%", "#0984e3"), use_container_width=True)
with r1_c2:
    st.plotly_chart(create_metric_chart("Server Temperature", "Temperature_C", "°C", "#d63031"), use_container_width=True)

r2_c1, r2_c2 = st.columns(2)
with r2_c1:
    st.plotly_chart(create_metric_chart("RAM Consumption", "RAM_Usage_%", "%", "#6c5ce7"), use_container_width=True)
with r2_c2:
    st.plotly_chart(create_metric_chart("Network Latency", "Network_Latency_ms", "ms", "#e17055"), use_container_width=True)

# Incident Table
st.markdown("---")
st.subheader("⚠️ Detected Incident Logs")
st.write("Timestamps where the system detected correlated metric anomalies:")

if len(incidents) > 0:
    st.dataframe(
        incidents[["Timestamp", "CPU_Load_%", "Temperature_C", "RAM_Usage_%", "Network_Latency_ms"]],
        use_container_width=True
    )
else:
    st.success("All systems operating within normal parameters.")