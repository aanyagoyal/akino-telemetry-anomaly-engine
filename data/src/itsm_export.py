import pandas as pd


class ITSMExporter:
    """
    Formats flagged incidents for import into enterprise ticketing systems.
    """

    PRIORITY_MAP = {
        "High CPU Throttling": "High",
        "Memory Saturation (Leak)": "High",
        "Latency Spike": "Medium",
        "Packet Drop / Collision (Possible DDoS)": "Highest",
        "Multi-variate Drift": "Medium",
        "Nominal": "Low",
    }

    @classmethod
    def _priority_for_cause(cls, root_cause: str) -> str:
        severity_order = ["Highest", "High", "Medium", "Low"]
        found = [
            cls.PRIORITY_MAP.get(part.strip(), "Medium")
            for part in root_cause.split("+")
        ]
        for level in severity_order:
            if level in found:
                return level
        return "Medium"

    @classmethod
    def to_jira_csv(cls, incidents_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in incidents_df.iterrows():
            root_cause = row.get("Root_Cause", "Multi-variate Drift")
            rows.append({
                "Summary": f"Infrastructure anomaly detected on Rack-Server-01 at {row['Timestamp']}",
                "Description": (
                    f"Root cause: {root_cause}\n"
                    f"CPU: {row.get('CPU_Load_%', 'N/A')}%, "
                    f"Temp: {row.get('Temperature_C', 'N/A')}C, "
                    f"RAM: {row.get('RAM_Usage_%', 'N/A')}%, "
                    f"Latency: {row.get('Network_Latency_ms', 'N/A')}ms, "
                    f"Packet Loss: {row.get('Packet_Loss_%', 'N/A')}%"
                ),
                "Priority": cls._priority_for_cause(root_cause),
                "Issue Type": "Incident",
                "Labels": "aiops,auto-detected",
            })
        return pd.DataFrame(rows)

    @classmethod
    def to_servicenow_csv(cls, incidents_df: pd.DataFrame) -> pd.DataFrame:
        jira_shaped = cls.to_jira_csv(incidents_df)
        return jira_shaped.rename(columns={
            "Summary": "Short description",
            "Description": "Description",
            "Priority": "Priority",
        })[["Short description", "Description", "Priority"]]