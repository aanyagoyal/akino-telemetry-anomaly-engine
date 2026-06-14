# Akino Connect - AIOps Infrastructure Anomaly Detection Engine

## Overview
An enterprise-grade, unsupervised machine learning telemetry monitoring pipeline developed during the Data Science Internship at Akino Connect. The engine continuously ingests and monitors multi-variate edge server and network telemetry to proactively isolate performance degradations, hardware choke points, and security anomalies before SLA thresholds are breached.

---

## Key Features
- **Multi-Variate Telemetry Ingestion**: Tracks and correlates `CPU_Load_%`, `RAM_Usage_%`, `Network_Latency_ms`, and `Packet_Loss_%` in real time.
- **Unsupervised Anomaly Isolation**: Utilizes an ensemble **Isolation Forest** algorithm calibrated for zero-label streaming infrastructure logs.
- **Automated Root Cause Diagnostics (RCA)**: Algorithmic heuristic layer that maps outlier vectors to real-world incidents (e.g., Memory Saturation/Leaks, DDoS spikes, Packet Drop collisions).
- **Multi-Model Benchmark Matrix**: Live comparative evaluation between Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM.
- **ITSM Incident Export**: One-click extraction of diagnostic audit logs formatted for enterprise ticketing systems (Jira/ServiceNow).

---

## Project Structure
```text
akino_internship_project/
│
├── data/
│   └── telemetry_generator.py   # Multi-variate telemetry simulation engine
│
├── src/
│   ├── __init__.py              # Package initializer
│   ├── model.py                 # Core ML models (Isolation Forest, LOF, SVM)
│   └── diagnostics.py           # Automated Root Cause Analysis (RCA)
│
├── app.py                       # Streamlit dashboard & orchestration UI
├── requirements.txt             # Production dependency specifications
└── README.md                    # System architecture documentation