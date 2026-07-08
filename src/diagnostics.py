import pandas as pd

class RootCauseDiagnostic:
    @staticmethod
    def identify_cause(row) -> str:
        if not row["Is_Incident"]:
            return "Nominal"
        
        causes = []
        if row["Temperature_C"] > 78:
            causes.append("Thermal Throttling (Overheat)")
        if row["CPU_Load_%"] > 75:
            causes.append("Compute Saturation")
        if row["RAM_Usage_%"] > 80:
            causes.append("Memory Saturation")
        if row["Network_Latency_ms"] > 100:
            causes.append("Latency Spike")
        if row["Packet_Loss_%"] > 5.0:
            causes.append("Network Packet Drop / Cable Fault")
            
        return " + ".join(causes) if causes else "Multi-metric Drift"

    @classmethod
    def annotate(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["Root_Cause"] = df.apply(cls.identify_cause, axis=1)
        return df