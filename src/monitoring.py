import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

os.makedirs("outputs", exist_ok=True)


class CreditModelMonitor:

    PSI_THRESHOLDS = {"stable": 0.10, "retrain": 0.25}

    def __init__(self, reference_scores, reference_features, model_name="credit_risk_v1"):
        self.reference_scores   = np.array(reference_scores)
        self.reference_features = reference_features.copy()
        self.model_name         = model_name
        self.alerts             = []
        self.timestamp          = datetime.now().isoformat()

    def psi(self, current_scores, n_bins=10):
        current = np.array(current_scores)

        bins       = np.percentile(self.reference_scores, np.linspace(0, 100, n_bins + 1))
        bins[0]    = -0.001
        bins[-1]   = 1.001
        bins       = np.unique(bins)

        ref_counts, _ = np.histogram(self.reference_scores, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        ref_pct = ref_counts / len(self.reference_scores) + 1e-10
        cur_pct = cur_counts / len(current) + 1e-10

        score = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        if score < self.PSI_THRESHOLDS["stable"]:
            status = "STABLE"
        elif score < self.PSI_THRESHOLDS["retrain"]:
            status = "MONITOR"
        else:
            status = "RETRAIN"
            self.alerts.append({
                "type": "PSI_ALERT", "value": score, "timestamp": self.timestamp
            })

        print(f"  PSI: {score:.4f}  [{status}]")
        return score

    def feature_drift(self, current_features, significance=0.01):
        rows = []

        for col in self.reference_features.select_dtypes(include=[np.number]).columns:
            if col not in current_features.columns:
                continue

            ref = self.reference_features[col].dropna().values
            cur = current_features[col].dropna().values

            if len(ref) == 0 or len(cur) == 0:
                continue

            ks_stat, p_val = stats.ks_2samp(ref, cur)
            rows.append({
                "feature":       col,
                "ks_stat":       round(float(ks_stat), 4),
                "p_value":       round(float(p_val), 6),
                "drifted":       bool(p_val < significance),
                "ref_mean":      round(float(ref.mean()), 4),
                "cur_mean":      round(float(cur.mean()), 4),
                "shift_pct":     round(float((cur.mean() - ref.mean()) / (abs(ref.mean()) + 1e-10) * 100), 1),
            })

        report = pd.DataFrame(rows).sort_values("ks_stat", ascending=False).reset_index(drop=True)
        n_drifted = report["drifted"].sum()

        print(f"  Feature drift: {n_drifted}/{len(report)} features drifted (p<{significance})")

        if n_drifted > 5:
            self.alerts.append({
                "type": "FEATURE_DRIFT", "n_drifted": int(n_drifted), "timestamp": self.timestamp
            })

        return report

    def performance_check(self, y_true, y_prob, auc_floor=0.68):
        auc  = float(roc_auc_score(y_true, y_prob))
        pr   = float(average_precision_score(y_true, y_prob))
        dr   = float(y_true.mean())

        status = "OK"
        if auc < auc_floor:
            status = "RETRAIN REQUIRED"
            self.alerts.append({
                "type": "PERFORMANCE_DEGRADATION", "roc_auc": auc, "timestamp": self.timestamp
            })

        print(f"  AUC: {auc:.4f}  PR-AUC: {pr:.4f}  Default rate: {dr:.1%}  [{status}]")
        return {"roc_auc": auc, "pr_auc": pr, "default_rate": dr, "status": status}

    def plot_dashboard(self, current_scores, current_features, drift_report):
        current_scores = np.array(current_scores)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Monitoring Dashboard — {self.model_name}", fontweight="bold")

        # Score distribution
        axes[0, 0].hist(self.reference_scores, bins=40, alpha=0.6, color="steelblue", label="Reference", density=True)
        axes[0, 0].hist(current_scores, bins=40, alpha=0.6, color="tomato", label="Current", density=True)
        axes[0, 0].set_title("Score Distribution")
        axes[0, 0].set_xlabel("Predicted Default Probability")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # KS bar chart
        if len(drift_report) > 0:
            top = drift_report.head(15).sort_values("ks_stat")
            colors = ["#e74c3c" if d else "#2ecc71" for d in top["drifted"]]
            axes[0, 1].barh(top["feature"], top["ks_stat"], color=colors, edgecolor="white")
            axes[0, 1].axvline(0.1, color="orange", linestyle="--", linewidth=1)
            axes[0, 1].set_title("Feature Drift (KS Statistic)")
            axes[0, 1].set_xlabel("KS Statistic")

        # Most drifted feature distribution
        if len(drift_report) > 0:
            top_feat = drift_report.iloc[0]["feature"]
            if top_feat in current_features.columns:
                ref_vals = self.reference_features[top_feat].dropna()
                cur_vals = current_features[top_feat].dropna()
                q01, q99 = ref_vals.quantile(0.01), ref_vals.quantile(0.99)
                axes[1, 0].hist(ref_vals.clip(q01, q99), bins=40, alpha=0.6, color="steelblue", label="Reference", density=True)
                axes[1, 0].hist(cur_vals.clip(q01, q99), bins=40, alpha=0.6, color="tomato", label="Current", density=True)
                axes[1, 0].set_title(f"Most Drifted: {top_feat}")
                axes[1, 0].legend()
                axes[1, 0].grid(alpha=0.3)

        # Alert panel
        if self.alerts:
            types = [a["type"] for a in self.alerts]
            unique, counts = np.unique(types, return_counts=True)
            axes[1, 1].bar(unique, counts, color="#e74c3c", edgecolor="white")
            axes[1, 1].set_title("Active Alerts")
            axes[1, 1].tick_params(axis="x", rotation=15)
        else:
            axes[1, 1].text(0.5, 0.5, "No Alerts\nModel Stable",
                ha="center", va="center", fontsize=16,
                color="green", fontweight="bold",
                transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Alert Status")

        plt.tight_layout()
        plt.savefig("outputs/drift_dashboard.png", dpi=150)
        plt.close()
        print("  Saved outputs/drift_dashboard.png")

    def should_retrain(self):
        critical = {"PSI_ALERT", "PERFORMANCE_DEGRADATION"}
        return any(a["type"] in critical for a in self.alerts)

    def summary(self):
        if not self.alerts:
            print("  No alerts. Model stable.")
            return
        for alert in self.alerts:
            print(f"  [ALERT] {alert['type']} — {alert}")


def run_monitoring(
    model_path="models/credit_risk_v1.pkl",
    data_path="data/processed/model_ready_dataset.parquet",
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run training.py first.")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}. Run feature_engineering.py first.")

    artifact = joblib.load(model_path)
    model    = artifact["model"]
    features = artifact["features"]

    df = pd.read_parquet(data_path)
    feat_cols = [c for c in features if c in df.columns]

    df_ref = df[df["split"] == "train"][feat_cols].fillna(0)
    df_cur = df[df["split"] == "test"][feat_cols].fillna(0)
    y_cur  = df[df["split"] == "test"]["target"]

    ref_scores = model.predict_proba(df_ref)[:, 1]
    cur_scores = model.predict_proba(df_cur)[:, 1]

    print(f"Reference: {len(df_ref):,} | Current: {len(df_cur):,}")

    monitor = CreditModelMonitor(ref_scores, df_ref, artifact["model_name"])

    print("\n[1] PSI Check:")
    monitor.psi(cur_scores)

    print("\n[2] Feature Drift:")
    drift_report = monitor.feature_drift(df_cur)

    print("\n[3] Performance:")
    monitor.performance_check(y_cur, cur_scores)

    print("\n[4] Dashboard:")
    monitor.plot_dashboard(cur_scores, df_cur, drift_report)

    print("\n[Summary]")
    monitor.summary()

    drift_report.to_csv("outputs/drift_report.csv", index=False)

    print(f"\nRetrain needed: {monitor.should_retrain()}")


if __name__ == "__main__":
    run_monitoring()
