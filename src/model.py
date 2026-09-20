import time

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "CPU_Load_%",
    "Temperature_C",
    "RAM_Usage_%",
    "Network_Latency_ms",
    "Packet_Loss_%",
]


class AnomalyEngine:
    """
    Primary detector used by the live dashboard. Uses Isolation Forest,
    now including Packet_Loss_% as a detection feature (previously only
    referenced in diagnostics.py but never actually fed to the model).
    """

    def __init__(self, contamination: float = 0.04):
        self.contamination = contamination
        self.feature_cols = FEATURE_COLS
        self.iso_model = IsolationForest(
            n_estimators=100, contamination=self.contamination, random_state=42
        )

    def run_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        features = data[self.feature_cols]

        data["Anomaly_Flag"] = self.iso_model.fit_predict(features)
        data["Is_Incident"] = data["Anomaly_Flag"] == -1
        data["Anomaly_Score"] = self.iso_model.decision_function(features).round(4)
        return data


class ModelBenchmark:
    """
    Runs Isolation Forest, Local Outlier Factor, and One-Class SVM on the
    same feature set and reports how they compare. This delivers the
    "Multi-Model Benchmark Matrix" the README advertises but that was never
    actually implemented.

    LOF and One-Class SVM are distance/density based, so features are
    standardized first -- Isolation Forest doesn't strictly need this but
    scaling doesn't hurt it either, and it keeps the comparison apples-to-apples.
    """

    def __init__(self, contamination: float = 0.04):
        self.contamination = contamination
        self.feature_cols = FEATURE_COLS
        self.scaler = StandardScaler()

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
            annotated_df: original df + one boolean "Is_Incident_<Model>" column per model
            summary_df: comparison table (rows = models, cols = count flagged, runtime, etc.)
        """
        data = df.copy()
        X = self.scaler.fit_transform(data[self.feature_cols])

        models = {
            "Isolation Forest": IsolationForest(
                n_estimators=100, contamination=self.contamination, random_state=42
            ),
            "Local Outlier Factor": LocalOutlierFactor(
                n_neighbors=20, contamination=self.contamination, novelty=False
            ),
            "One-Class SVM": OneClassSVM(nu=self.contamination, kernel="rbf", gamma="scale"),
        }

        summary_rows = []
        flags = {}

        for name, model in models.items():
            start = time.perf_counter()
            if name == "Local Outlier Factor":
                # LOF (non-novelty mode) only exposes fit_predict, no separate .fit()/.predict()
                preds = model.fit_predict(X)
            else:
                preds = model.fit_predict(X)
            elapsed_ms = (time.perf_counter() - start) * 1000

            is_incident = preds == -1
            flags[name] = is_incident
            summary_rows.append({
                "Model": name,
                "Anomalies Flagged": int(is_incident.sum()),
                "% of Records": round(100 * is_incident.mean(), 2),
                "Runtime (ms)": round(elapsed_ms, 2),
            })

        for name, is_incident in flags.items():
            col_name = f"Is_Incident_{name.replace(' ', '_')}"
            data[col_name] = is_incident

        # Agreement: how many models agree a given row is anomalous
        flag_matrix = pd.DataFrame(flags)
        data["Model_Agreement_Count"] = flag_matrix.sum(axis=1)

        summary_df = pd.DataFrame(summary_rows)
        return data, summary_df
