import pandas as pd
import numpy as np
from datetime import datetime


def generate_server_stream(n_samples=300, inject_fault=True):
    """
    Simulates multi-variate edge server / network telemetry.

    This is the single source of truth for synthetic telemetry generation.
    app.py imports this directly instead of maintaining its own copy, so the
    two no longer drift out of sync.
    """
    np.random.seed(42)
    timestamps = pd.date_range(end=datetime.now(), periods=n_samples, freq="min")

    # Normal metrics baseline
    cpu = 35 + 10 * np.sin(np.linspace(0, 12, n_samples)) + np.random.normal(0, 3, n_samples)
    temp = 55 + 5 * np.sin(np.linspace(0, 12, n_samples)) + np.random.normal(0, 2, n_samples)  # Server Temp in °C
    ram = 50 + 5 * np.linspace(0, 1.2, n_samples) + np.random.normal(0, 2, n_samples)
    latency = 20 + np.random.exponential(4, n_samples)
    # Packet loss: normally very low, exponentially distributed (long-tail spikes)
    packet_loss = np.random.exponential(0.3, n_samples)

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "CPU_Load_%": np.clip(cpu, 5, 100).round(2),
        "Temperature_C": np.clip(temp, 35, 105).round(2),
        "RAM_Usage_%": np.clip(ram, 10, 100).round(2),
        "Network_Latency_ms": np.clip(latency, 5, 400).round(2),
        "Packet_Loss_%": np.clip(packet_loss, 0, 100).round(2),
    })

    # Inject real-world failure spikes
    if inject_fault:
        # Event 1: Server Overheating & CPU Throttle (e.g. Fan Failure)
        idx1 = int(n_samples * 0.35)
        df.loc[idx1:idx1 + 5, "Temperature_C"] = np.clip(df.loc[idx1:idx1 + 5, "Temperature_C"] + 35, 40, 96)
        df.loc[idx1:idx1 + 5, "CPU_Load_%"] = np.clip(df.loc[idx1:idx1 + 5, "CPU_Load_%"] + 45, 0, 98)

        # Event 2: Network Congestion & High Latency
        idx2 = int(n_samples * 0.75)
        df.loc[idx2:idx2 + 6, "Network_Latency_ms"] += 180
        df.loc[idx2:idx2 + 6, "CPU_Load_%"] = np.clip(df.loc[idx2:idx2 + 6, "CPU_Load_%"] + 30, 0, 95)

        # Event 3: DDoS-style spike (high packet loss + latency, low-ish CPU signature)
        idx3 = int(n_samples * 0.55)
        df.loc[idx3:idx3 + 4, "Packet_Loss_%"] = np.clip(df.loc[idx3:idx3 + 4, "Packet_Loss_%"] + 8, 0, 100)
        df.loc[idx3:idx3 + 4, "Network_Latency_ms"] += 90

    return df
