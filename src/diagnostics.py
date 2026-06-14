import pandas as pd

class RootCauseDiagnostic:
    @staticmethod
    def identify_cause(row) -> str:
        if not row["Is_Incident"]:
            return "Nominal"
        causes = []
        if row["CPU_Load_%"] > 75:
            causes.append("High CPU Throttling")
        if row["RAM_Usage_%"] > 80:
            causes.append("Memory Saturation (Leak)")
        if row["Network_Latency_ms"] > 100:
            causes.append("Latency Spike")
        if row["Packet_Loss_%"] > 2.0:
            causes.append("Packet Drop / Collision")
        return " + ".join(causes) if causes else "Multi-variate Drift"

    @classmethod
    def annotate(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["Root_Cause"] = df.apply(cls.identify_cause, axis=1)
        return df