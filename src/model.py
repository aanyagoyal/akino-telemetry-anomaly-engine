from sklearn.ensemble import IsolationForest
import pandas as pd

class AnomalyEngine:
    def __init__(self, contamination=0.04):
        self.contamination = contamination
        self.feature_cols = ["CPU_Load_%", "Temperature_C", "RAM_Usage_%", "Network_Latency_ms"]
        self.iso_model = IsolationForest(n_estimators=100, contamination=self.contamination, random_state=42)

    def run_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        features = data[self.feature_cols]
        
        data["Anomaly_Flag"] = self.iso_model.fit_predict(features)
        data["Is_Incident"] = data["Anomaly_Flag"] == -1
        data["Anomaly_Score"] = self.iso_model.decision_function(features).round(4)
        return data