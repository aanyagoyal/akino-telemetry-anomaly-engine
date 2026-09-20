import pandas as pd


class RootCauseDiagnostic:
    """
    Rule-based heuristic layer that maps flagged anomaly rows to a
    human-readable probable cause. This is intentionally NOT ML-based --
    it's a fast, explainable annotation layer on top of whichever
    detector flagged the row.

    Note: previously referenced a "Packet_Loss_%" column that did not
    exist anywhere in the data pipeline, which meant calling this class
    would raise a KeyError. That column is now generated in
    data/telemetry_generator.py, so this is safe to use.
    """

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
        if row.get("Packet_Loss_%", 0) > 2.0:
            causes.append("Packet Drop / Collision (Possible DDoS)")

        return " + ".join(causes) if causes else "Multi-variate Drift"

    @classmethod
    def annotate(cls, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["Root_Cause"] = data.apply(cls.identify_cause, axis=1)
        return data
