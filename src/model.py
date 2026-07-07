import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

class AnomalyBenchmarkEngine:
    def __init__(self, contamination=0.04):
        self.contamination = contamination
        self.feature_cols = [
            "CPU_Load_%", 
            "Temperature_C", 
            "RAM_Usage_%", 
            "Network_Latency_ms", 
            "Packet_Loss_%"
        ]
        
    def evaluate_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM."""
        X = df[self.feature_cols]
        result_df = df.copy()

        # 1. Isolation Forest (Production Engine)
        iso = IsolationForest(contamination=self.contamination, random_state=42)
        result_df["Anomaly_IForest"] = iso.fit_predict(X)
        result_df["Score_IForest"] = iso.decision_function(X).round(4)

        # 2. Local Outlier Factor (Density Baseline)
        lof = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination)
        result_df["Anomaly_LOF"] = lof.fit_predict(X)

        # 3. One-Class SVM (Boundary Baseline)
        ocsvm = OneClassSVM(nu=self.contamination, kernel="rbf", gamma="scale")
        result_df["Anomaly_OCSVM"] = ocsvm.fit_predict(X)

        # Flagged if anomaly detected by Isolation Forest (-1)
        result_df["Anomaly_Flag"] = result_df["Anomaly_IForest"]
        result_df["Is_Incident"] = result_df["Anomaly_Flag"] == -1
        result_df["Anomaly_Score"] = result_df["Score_IForest"]

        return result_df

    def get_consensus_summary(self, evaluated_df: pd.DataFrame) -> dict:
        """Returns detection agreement across models."""
        total = len(evaluated_df)
        if_anomalies = (evaluated_df["Anomaly_IForest"] == -1).sum()
        lof_anomalies = (evaluated_df["Anomaly_LOF"] == -1).sum()
        ocsvm_anomalies = (evaluated_df["Anomaly_OCSVM"] == -1).sum()

        # Consensus: Points flagged by at least 2 models
        multi_flag = (
            (evaluated_df["Anomaly_IForest"] == -1).astype(int) +
            (evaluated_df["Anomaly_LOF"] == -1).astype(int) +
            (evaluated_df["Anomaly_OCSVM"] == -1).astype(int)
        ) >= 2
        consensus_count = multi_flag.sum()

        return {
            "Total_Samples": total,
            "Isolation_Forest_Flags": if_anomalies,
            "LOF_Flags": lof_anomalies,
            "OneClass_SVM_Flags": ocsvm_anomalies,
            "Ensemble_Consensus_Flags": consensus_count
        }