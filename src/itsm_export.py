import pandas as pd


class ITSMExporter:


    PRIORITY_MAP = {
        "High CPU Throttling": "High",
        "Memory Saturation (Leak)": "High",
        "Latency Spike": "Medium",
        "Packet Drop / Collision (Possible DDoS)": "Highest",
        "Multi-variate Drift": "Medium",
        "Nominal": "Low",
    }

    # Short, ticket-friendly label per cause component, plus the metric
    # threshold that triggered it so the description only reports what's
    # actually anomalous instead of dumping every metric on every row.
    CAUSE_META = {
        "High CPU Throttling": {
            "label": "cpu-throttle",
            "metric": "CPU_Load_%", "unit": "%", "threshold": "> 75%",
        },
        "Memory Saturation (Leak)": {
            "label": "memory-leak",
            "metric": "RAM_Usage_%", "unit": "%", "threshold": "> 80%",
        },
        "Latency Spike": {
            "label": "latency-spike",
            "metric": "Network_Latency_ms", "unit": "ms", "threshold": "> 100ms",
        },
        "Packet Drop / Collision (Possible DDoS)": {
            "label": "packet-loss-ddos-suspected",
            "metric": "Packet_Loss_%", "unit": "%", "threshold": "> 2.0%",
        },
        "Multi-variate Drift": {
            "label": "multivariate-drift",
            "metric": None, "unit": None, "threshold": None,
        },
    }

    @classmethod
    def _cause_parts(cls, root_cause: str) -> list:
        return [part.strip() for part in root_cause.split("+")]

    @classmethod
    def _priority_for_cause(cls, root_cause: str) -> str:
        # Root_Cause can be a combination like "High CPU Throttling + Latency Spike"
        # Take the highest-severity component present.
        severity_order = ["Highest", "High", "Medium", "Low"]
        found = [
            cls.PRIORITY_MAP.get(part, "Medium")
            for part in cls._cause_parts(root_cause)
        ]
        for level in severity_order:
            if level in found:
                return level
        return "Medium"

    @classmethod
    def _issue_type_for_cause(cls, root_cause: str) -> str:
        # A suspected DDoS is a security concern, not just an ops incident --
        # route it differently so it doesn't get triaged like a CPU spike.
        if "Packet Drop / Collision (Possible DDoS)" in cls._cause_parts(root_cause):
            return "Security Incident"
        return "Incident"

    @classmethod
    def _labels_for_cause(cls, root_cause: str) -> str:
        base = ["aiops", "auto-detected"]
        for part in cls._cause_parts(root_cause):
            meta = cls.CAUSE_META.get(part)
            if meta:
                base.append(meta["label"])
        return ",".join(base)

    @classmethod
    def _summary_for_cause(cls, root_cause: str, hostname: str, timestamp) -> str:
        headline = root_cause if root_cause != "Multi-variate Drift" else "Multi-variate anomaly"
        return f"{headline} detected on {hostname} at {timestamp}"

    @classmethod
    def _description_for_row(cls, row, root_cause: str) -> str:
        lines = [f"Root cause: {root_cause}"]
        breached = []
        for part in cls._cause_parts(root_cause):
            meta = cls.CAUSE_META.get(part)
            if meta and meta["metric"]:
                value = row.get(meta["metric"], "N/A")
                if isinstance(value, (int, float)):
                    value = round(value, 2)
                breached.append(f"{meta['metric']}: {value}{meta['unit']} (threshold {meta['threshold']})")
        if breached:
            lines.append("Breached metrics:")
            lines.extend(f"  - {b}" for b in breached)
        lines.append(
            "Full snapshot -- CPU: {cpu}%, Temp: {temp}C, RAM: {ram}%, "
            "Latency: {lat}ms, Packet Loss: {pl}%".format(
                cpu=row.get("CPU_Load_%", "N/A"),
                temp=row.get("Temperature_C", "N/A"),
                ram=row.get("RAM_Usage_%", "N/A"),
                lat=row.get("Network_Latency_ms", "N/A"),
                pl=row.get("Packet_Loss_%", "N/A"),
            )
        )
        return "\n".join(lines)

    @classmethod
    def to_jira_csv(cls, incidents_df: pd.DataFrame, hostname: str = "Rack-Server-01") -> pd.DataFrame:
        """Returns a DataFrame shaped for Jira's CSV importer."""
        rows = []
        for _, row in incidents_df.iterrows():
            root_cause = row.get("Root_Cause", "Multi-variate Drift")
            rows.append({
                "Summary": cls._summary_for_cause(root_cause, hostname, row["Timestamp"]),
                "Description": cls._description_for_row(row, root_cause),
                "Priority": cls._priority_for_cause(root_cause),
                "Issue Type": cls._issue_type_for_cause(root_cause),
                "Labels": cls._labels_for_cause(root_cause),
            })
        return pd.DataFrame(rows)

    @classmethod
    def to_servicenow_csv(cls, incidents_df: pd.DataFrame, hostname: str = "Rack-Server-01") -> pd.DataFrame:
        """Returns a DataFrame shaped for ServiceNow incident import."""
        jira_shaped = cls.to_jira_csv(incidents_df, hostname=hostname)
        return jira_shaped.rename(columns={
            "Summary": "Short description",
            "Description": "Description",
            "Priority": "Priority",
        })[["Short description", "Description", "Priority"]]
