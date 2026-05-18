"""
src/monitoring.py
=================
Phase 5 - Model Drift Detection and Monitoring

How to run:
    python src/monitoring.py

What it does:
    - Population Stability Index (score drift)
    - Kolmogorov-Smirnov test (feature drift)
    - Performance degradation alerts
    - Auto retraining trigger logic
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

os.makedirs('outputs',  exist_ok=True)
os.makedirs('models',   exist_ok=True)

# ============================================================
# PSI + KS MONITOR
# ============================================================
class CreditModelMonitor:
    """
    Monitors deployed credit risk model for degradation.

    Three monitoring layers:
    1. Score PSI      - did predicted probabilities shift?
    2. Feature KS     - did input distributions shift?
    3. Performance    - did AUC / default rate change?

    PSI thresholds (industry standard):
        < 0.10  GREEN  - No action needed
        0.10-0.25 YELLOW - Monitor closely
        > 0.25  RED    - Retrain immediately
    """

    def __init__(
        self,
        reference_scores  : np.ndarray,
        reference_features: pd.DataFrame,
        model_name        : str = 'credit_risk_v1',
    ):
        self.reference_scores   = np.array(reference_scores)
        self.reference_features = reference_features.copy()
        self.model_name         = model_name
        self.alert_log          = []
        self.check_timestamp    = datetime.now().isoformat()

    # ── PSI ───────────────────────────────────────────────────
    def population_stability_index(
        self,
        current_scores: np.ndarray,
        n_bins        : int = 10,
        verbose       : bool = True,
    ) -> float:
        """
        Measure how much the score distribution has shifted.

        Formula:
        PSI = sum((Current% - Reference%) * ln(Current% / Reference%))

        Args:
            current_scores: predicted probabilities from new data
            n_bins: number of buckets (10 is standard)

        Returns:
            PSI value (float)
        """
        current_scores = np.array(current_scores)

        # Create bins from reference distribution
        bins       = np.percentile(self.reference_scores, np.linspace(0, 100, n_bins + 1))
        bins[0]    = -0.001
        bins[-1]   = 1.001
        bins       = np.unique(bins)  # remove duplicates

        ref_counts, _ = np.histogram(self.reference_scores, bins=bins)
        cur_counts, _ = np.histogram(current_scores, bins=bins)

        # Convert to percentages (avoid div by zero)
        ref_pct = ref_counts / len(self.reference_scores) + 1e-10
        cur_pct = cur_counts / len(current_scores)        + 1e-10

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        # Classify
        if psi < 0.10:
            status = 'STABLE'
            color  = 'GREEN'
        elif psi < 0.25:
            status = 'MONITOR'
            color  = 'YELLOW'
        else:
            status = 'RETRAIN'
            color  = 'RED'

        if verbose:
            print(f'  PSI Score   : {psi:.4f}')
            print(f'  Status      : [{color}] {status}')
            if status == 'RETRAIN':
                self.alert_log.append({
                    'type'     : 'PSI_ALERT',
                    'value'    : psi,
                    'status'   : status,
                    'timestamp': self.check_timestamp,
                })

        return psi

    # ── KS Feature Drift ──────────────────────────────────────
    def feature_drift_report(
        self,
        current_features  : pd.DataFrame,
        significance_level: float = 0.01,
        top_n             : int   = 20,
    ) -> pd.DataFrame:
        """
        Run Kolmogorov-Smirnov test on each numeric feature.

        KS test compares two distributions without assuming normality.
        p_value < 0.01 means the distributions are significantly different.

        Args:
            current_features: new data features
            significance_level: p-value threshold for drift detection
            top_n: show top N drifted features

        Returns:
            DataFrame with KS stats per feature
        """
        results = []

        numeric_cols = self.reference_features.select_dtypes(
            include=[np.number]
        ).columns

        for col in numeric_cols:
            if col not in current_features.columns:
                continue

            ref_vals = self.reference_features[col].dropna().values
            cur_vals = current_features[col].dropna().values

            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)

            results.append({
                'feature'        : col,
                'ks_statistic'   : round(float(ks_stat), 4),
                'p_value'        : round(float(p_value), 6),
                'drift_detected' : bool(p_value < significance_level),
                'ref_mean'       : round(float(ref_vals.mean()), 4),
                'cur_mean'       : round(float(cur_vals.mean()), 4),
                'mean_shift_pct' : round(
                    float((cur_vals.mean() - ref_vals.mean()) /
                    (abs(ref_vals.mean()) + 1e-10) * 100), 1
                ),
            })

        df_results = (
            pd.DataFrame(results)
            .sort_values('ks_statistic', ascending=False)
            .reset_index(drop=True)
        )

        n_drifted = df_results['drift_detected'].sum()
        print(f'\n  Feature Drift Report (KS Test, p<{significance_level}):')
        print(f'  Drifted features: {n_drifted} / {len(df_results)}')
        print(f'\n  {"Feature":<35} {"KS":>6} {"p-val":>10} {"Drift":>7} {"Ref Mean":>10} {"Cur Mean":>10} {"Shift%":>8}')
        print(f'  {"-"*35} {"-"*6} {"-"*10} {"-"*7} {"-"*10} {"-"*10} {"-"*8}')

        for _, row in df_results.head(top_n).iterrows():
            drift_flag = 'DRIFT' if row['drift_detected'] else 'ok'
            print(
                f'  {row["feature"]:<35} '
                f'{row["ks_statistic"]:>6.4f} '
                f'{row["p_value"]:>10.6f} '
                f'{drift_flag:>7} '
                f'{row["ref_mean"]:>10.2f} '
                f'{row["cur_mean"]:>10.2f} '
                f'{row["mean_shift_pct"]:>7.1f}%'
            )

        if n_drifted > 5:
            self.alert_log.append({
                'type'      : 'FEATURE_DRIFT_ALERT',
                'n_drifted' : int(n_drifted),
                'timestamp' : self.check_timestamp,
            })

        return df_results

    # ── Performance Check ─────────────────────────────────────
    def performance_check(
        self,
        y_true       : pd.Series,
        y_prob       : np.ndarray,
        auc_threshold: float = 0.68,
    ) -> dict:
        """
        Evaluate model performance on recent labeled data.
        Labels are only available 90+ days after loan origination.

        Args:
            y_true: actual outcomes (0/1)
            y_prob: model predicted probabilities
            auc_threshold: minimum acceptable AUC before alert

        Returns:
            dict with roc_auc, pr_auc, and alert status
        """
        auc    = float(roc_auc_score(y_true, y_prob))
        prauc  = float(average_precision_score(y_true, y_prob))
        dr     = float(y_true.mean())

        status = 'OK'
        if auc < auc_threshold:
            status = 'ALERT: RETRAIN REQUIRED'
            self.alert_log.append({
                'type'     : 'PERFORMANCE_DEGRADATION',
                'roc_auc'  : auc,
                'threshold': auc_threshold,
                'timestamp': self.check_timestamp,
            })

        print(f'\n  Performance Check:')
        print(f'  ROC-AUC      : {auc:.4f}  (threshold: {auc_threshold:.2f})')
        print(f'  PR-AUC       : {prauc:.4f}')
        print(f'  Default Rate : {dr:.1%}')
        print(f'  Status       : {status}')

        return {
            'roc_auc'     : round(auc, 4),
            'pr_auc'      : round(prauc, 4),
            'default_rate': round(dr, 4),
            'status'      : status,
            'timestamp'   : self.check_timestamp,
        }

    # ── Plot Drift Dashboard ──────────────────────────────────
    def plot_drift_dashboard(
        self,
        current_scores  : np.ndarray,
        current_features: pd.DataFrame,
        drift_report    : pd.DataFrame,
        output_path     : str = 'outputs/drift_dashboard.png',
    ) -> None:
        """Generate monitoring dashboard chart."""
        current_scores = np.array(current_scores)
        fig, axes      = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Model Monitoring Dashboard — {self.model_name}\n'
            f'Checked: {self.check_timestamp[:10]}',
            fontsize=13, fontweight='bold'
        )

        # Plot 1: Score distribution comparison
        axes[0,0].hist(
            self.reference_scores, bins=40,
            alpha=0.6, color='steelblue',
            label='Reference (train)', density=True
        )
        axes[0,0].hist(
            current_scores, bins=40,
            alpha=0.6, color='tomato',
            label='Current (new)', density=True
        )
        axes[0,0].set_title('Score Distribution Shift', fontweight='bold')
        axes[0,0].set_xlabel('Predicted Default Probability')
        axes[0,0].set_ylabel('Density')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Plot 2: KS statistic bar chart
        if len(drift_report) > 0:
            top_drift = drift_report.head(15).sort_values('ks_statistic')
            bar_colors = [
                '#e74c3c' if d else '#2ecc71'
                for d in top_drift['drift_detected']
            ]
            axes[0,1].barh(
                top_drift['feature'],
                top_drift['ks_statistic'],
                color=bar_colors, edgecolor='white'
            )
            axes[0,1].axvline(0.1, color='orange', linestyle='--', label='KS=0.10')
            axes[0,1].set_title(
                'Feature Drift (KS Statistic)\nRed=Drifted | Green=Stable',
                fontweight='bold'
            )
            axes[0,1].set_xlabel('KS Statistic')
            axes[0,1].legend(fontsize=8)

        # Plot 3: Feature distribution for most drifted feature
        if len(drift_report) > 0:
            top_feat = drift_report.iloc[0]['feature']
            if top_feat in current_features.columns:
                ref_vals = self.reference_features[top_feat].dropna()
                cur_vals = current_features[top_feat].dropna()
                axes[1,0].hist(
                    ref_vals.clip(
                        ref_vals.quantile(0.01),
                        ref_vals.quantile(0.99)
                    ),
                    bins=40, alpha=0.6, color='steelblue',
                    label='Reference', density=True
                )
                axes[1,0].hist(
                    cur_vals.clip(
                        cur_vals.quantile(0.01),
                        cur_vals.quantile(0.99)
                    ),
                    bins=40, alpha=0.6, color='tomato',
                    label='Current', density=True
                )
                axes[1,0].set_title(
                    f'Most Drifted Feature: {top_feat}',
                    fontweight='bold'
                )
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)

        # Plot 4: Alert log summary
        if self.alert_log:
            alert_types = [a['type'] for a in self.alert_log]
            unique, counts = np.unique(alert_types, return_counts=True)
            colors = ['#e74c3c', '#f39c12', '#e67e22'][:len(unique)]
            axes[1,1].bar(unique, counts, color=colors, edgecolor='white')
            axes[1,1].set_title('Active Alerts', fontweight='bold')
            axes[1,1].set_ylabel('Count')
            axes[1,1].tick_params(axis='x', rotation=15)
        else:
            axes[1,1].text(
                0.5, 0.5, 'No Alerts\nModel Stable',
                ha='center', va='center',
                fontsize=16, color='green', fontweight='bold',
                transform=axes[1,1].transAxes
            )
            axes[1,1].set_title('Alert Status', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  [SAVED] {output_path}')

    # ── Print Alert Summary ───────────────────────────────────
    def print_alert_summary(self) -> None:
        """Print all active alerts."""
        print(f'\n  ALERT SUMMARY ({len(self.alert_log)} alerts):')
        print('  ' + '=' * 50)
        if not self.alert_log:
            print('  [GREEN] No alerts. Model is stable.')
        for alert in self.alert_log:
            print(f'  [RED] {alert["type"]}')
            for k, v in alert.items():
                if k != 'type':
                    print(f'        {k}: {v}')

    # ── Retraining Decision ───────────────────────────────────
    def should_retrain(self) -> bool:
        """Return True if any critical alert was triggered."""
        critical = ['PSI_ALERT', 'PERFORMANCE_DEGRADATION']
        for alert in self.alert_log:
            if alert['type'] in critical:
                return True
        return False


# ============================================================
# RETRAINING TRIGGERS
# ============================================================
RETRAINING_TRIGGERS = {
    'PSI > 0.25'              : 'Score distribution shifted significantly',
    'ROC-AUC < 0.68'         : 'Model discrimination power degraded',
    'Top-5 feature drift'     : 'Critical predictor distributions changed',
    'Default rate deviation>3%': 'Actuals diverging from predictions',
    'Quarterly calendar'      : 'Mandatory retraining every 90 days',
}


# ============================================================
# MAIN — Demo run with synthetic data
# ============================================================
def run_monitoring_demo(
    model_path  : str = 'models/credit_risk_v1.pkl',
    data_path   : str = 'data/processed/model_ready_dataset.parquet',
) -> None:
    """
    Demonstrate monitoring with reference vs simulated new data.
    In production replace simulated data with real new loan applications.
    """
    print('=' * 60)
    print('  CREDIT RISK — MODEL MONITORING')
    print('=' * 60)

    # Load model
    if not os.path.exists(model_path):
        print(f'[ERROR] Model not found: {model_path}')
        print('  Run python src/training.py first')
        return

    artifact = joblib.load(model_path)
    model    = artifact['model']
    features = artifact['feature_names']
    print(f'\n[OK] Model loaded: {artifact["model_name"]}')
    print(f'[OK] Saved AUC   : {artifact["test_roc_auc"]:.4f}')

    # Load reference data
    if not os.path.exists(data_path):
        print(f'[ERROR] Data not found: {data_path}')
        return

    df = pd.read_parquet(data_path)
    print(f'[OK] Data loaded : {df.shape}')

    # Use training data as reference
    drop_cols = ['target', 'split', 'issue_year']
    feat_cols = [c for c in features if c in df.columns]

    if 'issue_year' in df.columns:
        ref_mask = df['issue_year'] <= 2015
        cur_mask = df['issue_year'] >= 2017
    else:
        mid      = len(df) // 2
        ref_mask = pd.Series([True]*mid  + [False]*(len(df)-mid), index=df.index)
        cur_mask = pd.Series([False]*mid + [True]*(len(df)-mid),  index=df.index)

    df_ref = df[ref_mask][feat_cols].fillna(0)
    df_cur = df[cur_mask][feat_cols].fillna(0)
    y_cur  = df[cur_mask]['target']

    # Predict scores
    ref_scores = model.predict_proba(df_ref)[:, 1]
    cur_scores = model.predict_proba(df_cur)[:, 1]

    print(f'\n  Reference data : {len(df_ref):,} loans')
    print(f'  Current data   : {len(df_cur):,} loans')

    # ── Initialize monitor ────────────────────────────────────
    monitor = CreditModelMonitor(
        reference_scores   = ref_scores,
        reference_features = df_ref,
        model_name         = artifact['model_name'],
    )

    # ── Run checks ────────────────────────────────────────────
    print('\n[CHECK 1] Population Stability Index (PSI):')
    psi = monitor.population_stability_index(cur_scores)

    print('\n[CHECK 2] Feature Drift (KS Test):')
    drift_report = monitor.feature_drift_report(df_cur)

    print('\n[CHECK 3] Performance Check:')
    perf = monitor.performance_check(y_cur, cur_scores)

    # ── Plot dashboard ────────────────────────────────────────
    print('\n[PLOT] Generating monitoring dashboard...')
    monitor.plot_drift_dashboard(
        current_scores   = cur_scores,
        current_features = df_cur,
        drift_report     = drift_report,
        output_path      = 'outputs/drift_dashboard.png',
    )

    # ── Alert summary ─────────────────────────────────────────
    monitor.print_alert_summary()

    # ── Retraining decision ───────────────────────────────────
    print(f'\n[DECISION] Should retrain? {monitor.should_retrain()}')
    if monitor.should_retrain():
        print('  Action: Schedule retraining pipeline')
    else:
        print('  Action: Continue monitoring. Next check in 7 days.')

    # ── Save drift report ─────────────────────────────────────
    drift_path = 'outputs/drift_report.csv'
    drift_report.to_csv(drift_path, index=False)
    print(f'\n[SAVED] {drift_path}')

    print('\n' + '=' * 60)
    print('  MONITORING COMPLETE!')
    print('=' * 60)


if __name__ == '__main__':
    run_monitoring_demo()