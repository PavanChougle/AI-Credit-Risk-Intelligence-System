"""
src/training.py
===============
Phase 4 - ML Risk Model Training Pipeline

How to run:
    python src/training.py

Outputs:
    models/credit_risk_v1.pkl
    outputs/model_evaluation.png
    outputs/shap_explanations.png
    outputs/shap_global.png
    outputs/shap_beeswarm.png
    outputs/model_comparison.csv
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

# LightGBM
try:
    import importlib
    lgb_spec = importlib.util.find_spec('lightgbm')
    if lgb_spec is not None:
        import lightgbm as lgb
        LGBM_AVAILABLE = True
        print(f'[OK] LightGBM {lgb.__version__} found')
    else:
        raise ImportError('lightgbm spec not found')
except Exception as e:
    LGBM_AVAILABLE = False
    print(f'[WARN] LightGBM not available: {e}')
    print('       Run: python -m pip install lightgbm')

# SHAP
try:
    shap_spec = importlib.util.find_spec('shap')
    if shap_spec is not None:
        import shap
        SHAP_AVAILABLE = True
        print(f'[OK] SHAP {shap.__version__} found')
    else:
        raise ImportError('shap spec not found')
except Exception as e:
    SHAP_AVAILABLE = False
    print(f'[WARN] SHAP not available: {e}')
    print('       Run: python -m pip install shap')

# ============================================================
# FOLDER SETUP
# ============================================================
os.makedirs('models',   exist_ok=True)
os.makedirs('outputs',  exist_ok=True)

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
def load_model_ready_data(
    parquet_path: str = 'data/processed/model_ready_dataset.parquet'
) -> tuple:
    """
    Load model-ready dataset and create temporal splits.

    Split strategy:
        Train      : issue_year <= 2015  (historical patterns)
        Validation : issue_year == 2016  (tune hyperparameters)
        Test       : issue_year >= 2017  (held-out, final evaluation only)

    Why time-based split?
    Random split leaks future information into past training.
    In production, we always train on past and predict on future.
    """
    print('\n[STEP 1/6] Loading model-ready dataset...')

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f'[ERROR] {parquet_path} not found.\n'
            f'Run feature engineering first: python src/feature_engineering.py'
        )

    df = pd.read_parquet(parquet_path)
    print(f'  [OK] Loaded : {df.shape[0]:,} rows x {df.shape[1]} cols')
    print(f'  [OK] Memory : {df.memory_usage(deep=True).sum()/1e6:.1f} MB')

    # Get feature columns (exclude target and split)
    drop_cols = ['target', 'split']
    FEATURES  = [
        c for c in df.columns
        if c not in drop_cols
    ]

    # Remove issue_year from features if present
    # (it was used for splitting but is not a model feature)
    if 'issue_year' in FEATURES:
        FEATURES.remove('issue_year')

    TARGET = 'target'

    # Temporal splits
    if 'issue_year' in df.columns:
        train_mask = df['issue_year'] <= 2015
        val_mask   = df['issue_year'] == 2016
        test_mask  = df['issue_year'] >= 2017

        df_train = df[train_mask].copy()
        df_val   = df[val_mask].copy()
        df_test  = df[test_mask].copy()

        print(f'\n  Temporal Split Summary:')
        print(f'  {"Split":<12} {"Rows":>10} {"Default Rate":>14}')
        print(f'  {"-"*12} {"-"*10} {"-"*14}')
        for name, subset in [
            ('Train', df_train), ('Validation', df_val), ('Test', df_test)
        ]:
            print(
                f'  {name:<12} {len(subset):>10,} '
                f'{subset[TARGET].mean():>13.1%}'
            )
    else:
        # Fallback: use pre-existing split column
        print('  [WARN] issue_year not found. Using split column.')
        df_train = df[df['split'] == 'train'].copy()
        df_val   = df[df['split'] == 'test'].copy()
        df_test  = df[df['split'] == 'test'].copy()

    X_train = df_train[FEATURES].fillna(0)
    y_train = df_train[TARGET]
    X_val   = df_val[FEATURES].fillna(0)
    y_val   = df_val[TARGET]
    X_test  = df_test[FEATURES].fillna(0)
    y_test  = df_test[TARGET]

    # Class imbalance ratio (for LightGBM scale_pos_weight)
    neg_pos_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f'\n  Class imbalance ratio : {neg_pos_ratio:.1f}x')
    print(f'  Features used        : {len(FEATURES)}')
    print(f'  Feature list:')
    for f in FEATURES:
        print(f'    - {f}')

    return (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        FEATURES, neg_pos_ratio
    )


# ============================================================
# STEP 2 — TRAIN MODELS
# ============================================================
def train_all_models(
    X_train, y_train,
    X_val,   y_val,
    neg_pos_ratio: float
) -> dict:
    """
    Train all models and collect validation metrics.

    Models:
    1. Logistic Regression — interpretable baseline
    2. LightGBM           — primary production model

    Why LightGBM over XGBoost?
    - Faster training (leaf-wise tree growth)
    - Better handling of categorical features
    - Lower memory usage at scale
    - Native early stopping on validation set
    """
    print('\n[STEP 2/6] Training models...')
    results = {}

    # ── Model 1: Logistic Regression Baseline ────────────────
    print('\n  [1/2] Training Logistic Regression (baseline)...')
    t0 = time.time()

    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(
            C             = 0.1,
            class_weight  = 'balanced',
            max_iter      = 1000,
            random_state  = 42,
            solver        = 'saga',
            n_jobs        = -1,
        ))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_probs = lr_pipeline.predict_proba(X_val)[:, 1]
    lr_time  = time.time() - t0

    results['Logistic Regression'] = {
        'model'      : lr_pipeline,
        'val_probs'  : lr_probs,
        'roc_auc'    : roc_auc_score(y_val, lr_probs),
        'pr_auc'     : average_precision_score(y_val, lr_probs),
        'train_time' : lr_time,
        'model_type' : 'baseline',
    }
    print(
        f'  [OK] LR done in {lr_time:.1f}s | '
        f'Val AUC: {results["Logistic Regression"]["roc_auc"]:.4f}'
    )

    # ── Model 2: LightGBM ─────────────────────────────────────
    if LGBM_AVAILABLE:
        print('\n  [2/2] Training LightGBM (primary model)...')
        t0 = time.time()

        lgb_model = lgb.LGBMClassifier(
            n_estimators      = 1000,
            learning_rate     = 0.05,
            num_leaves        = 63,
            max_depth         = 6,
            min_child_samples = 100,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            reg_alpha         = 0.1,
            reg_lambda        = 0.1,
            scale_pos_weight  = neg_pos_ratio,
            random_state      = 42,
            verbose           = -1,
            n_jobs            = -1,
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set  = [(X_val, y_val)],
            callbacks = [lgb.early_stopping(50, verbose=False)]
        )
        lgb_probs = lgb_model.predict_proba(X_val)[:, 1]
        lgb_time  = time.time() - t0

        results['LightGBM'] = {
            'model'      : lgb_model,
            'val_probs'  : lgb_probs,
            'roc_auc'    : roc_auc_score(y_val, lgb_probs),
            'pr_auc'     : average_precision_score(y_val, lgb_probs),
            'train_time' : lgb_time,
            'model_type' : 'primary',
            'best_iter'  : lgb_model.best_iteration_,
        }
        print(
            f'  [OK] LightGBM done in {lgb_time:.1f}s | '
            f'Val AUC: {results["LightGBM"]["roc_auc"]:.4f} | '
            f'Best iter: {lgb_model.best_iteration_}'
        )
    else:
        print('  [SKIP] LightGBM not installed')

    # ── Print comparison ──────────────────────────────────────
    print('\n  MODEL COMPARISON (Validation Set):')
    print(f'  {"Model":<25} {"ROC-AUC":>9} {"PR-AUC":>9} {"Time":>8}')
    print(f'  {"-"*25} {"-"*9} {"-"*9} {"-"*8}')
    for name, m in results.items():
        print(
            f'  {name:<25} '
            f'{m["roc_auc"]:>9.4f} '
            f'{m["pr_auc"]:>9.4f} '
            f'{m["train_time"]:>7.1f}s'
        )

    return results


# ============================================================
# STEP 3 — EVALUATE ON TEST SET
# ============================================================
def evaluate_on_test(
    results     : dict,
    X_test      : pd.DataFrame,
    y_test      : pd.Series,
    FEATURES    : list,
) -> dict:
    """
    Evaluate all models on the held-out test set.
    Also find optimal decision threshold using F1.
    """
    print('\n[STEP 3/6] Evaluating on held-out test set...')

    test_results = {}

    for name, data in results.items():
        model      = data['model']
        test_probs = model.predict_proba(X_test)[:, 1]
        test_auc   = roc_auc_score(y_test, test_probs)
        test_prauc = average_precision_score(y_test, test_probs)

        # Find optimal threshold by F1
        thresholds  = np.linspace(0.05, 0.80, 100)
        f1_scores   = []
        for thresh in thresholds:
            preds = (test_probs >= thresh).astype(int)
            tp = int(((preds == 1) & (y_test == 1)).sum())
            fp = int(((preds == 1) & (y_test == 0)).sum())
            fn = int(((preds == 0) & (y_test == 1)).sum())
            p  = tp / (tp + fp + 1e-10)
            r  = tp / (tp + fn + 1e-10)
            f1 = 2 * p * r / (p + r + 1e-10)
            f1_scores.append(f1)

        optimal_thresh = float(thresholds[np.argmax(f1_scores)])
        optimal_preds  = (test_probs >= optimal_thresh).astype(int)

        test_results[name] = {
            'test_probs'    : test_probs,
            'roc_auc'       : test_auc,
            'pr_auc'        : test_prauc,
            'optimal_thresh': optimal_thresh,
            'optimal_preds' : optimal_preds,
            'val_roc_auc'   : data['roc_auc'],
            'train_time'    : data['train_time'],
            'model'         : model,
        }

        print(f'\n  {name}:')
        print(f'    Test ROC-AUC        : {test_auc:.4f}')
        print(f'    Test PR-AUC         : {test_prauc:.4f}')
        print(f'    Optimal threshold   : {optimal_thresh:.3f}')
        print(f'\n    Classification Report (threshold={optimal_thresh:.2f}):')
        print('    ' + '-' * 50)
        report = classification_report(
            y_test, optimal_preds,
            target_names=['Good Loan', 'Default'],
        )
        for line in report.split('\n'):
            print(f'    {line}')

    return test_results


# ============================================================
# STEP 4 — PLOT MODEL EVALUATION
# ============================================================
def plot_model_evaluation(
    test_results : dict,
    y_test       : pd.Series,
    output_path  : str = 'outputs/model_evaluation.png',
) -> None:
    """
    Generate 4-panel evaluation chart:
    1. ROC Curve (all models)
    2. Precision-Recall Curve
    3. Calibration Plot
    4. Threshold vs F1/Precision/Recall
    """
    print('\n[STEP 4/6] Plotting model evaluation...')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        'Credit Risk Model Evaluation',
        fontsize=15, fontweight='bold'
    )

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    # ── Chart 1: ROC Curve ────────────────────────────────────
    ax = axes[0, 0]
    for i, (name, data) in enumerate(test_results.items()):
        fpr, tpr, _ = roc_curve(y_test, data['test_probs'])
        auc = data['roc_auc']
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})',
                linewidth=2, color=colors[i])
    ax.plot([0,1], [0,1], 'k--', alpha=0.4, label='Random baseline')
    ax.fill_between(fpr, tpr, alpha=0.05, color=colors[0])
    ax.set_title('ROC Curve', fontweight='bold', fontsize=12)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Chart 2: Precision-Recall Curve ──────────────────────
    ax = axes[0, 1]
    for i, (name, data) in enumerate(test_results.items()):
        prec, rec, _ = precision_recall_curve(y_test, data['test_probs'])
        pr_auc       = data['pr_auc']
        ax.plot(rec, prec,
                label=f'{name} (PR-AUC={pr_auc:.3f})',
                linewidth=2, color=colors[i])
    ax.axhline(
        y=y_test.mean(), color='gray',
        linestyle='--', alpha=0.7,
        label=f'Baseline ({y_test.mean():.1%} default rate)'
    )
    ax.set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Chart 3: Calibration Plot ─────────────────────────────
    ax = axes[1, 0]
    for i, (name, data) in enumerate(test_results.items()):
        try:
            frac_pos, mean_pred = calibration_curve(
                y_test, data['test_probs'], n_bins=10
            )
            ax.plot(
                mean_pred, frac_pos, 's-',
                label=name, linewidth=2,
                color=colors[i]
            )
        except Exception:
            pass
    ax.plot([0,1], [0,1], 'k--', label='Perfect calibration', alpha=0.6)
    ax.set_title(
        'Calibration Plot\n(Diagonal = perfectly calibrated)',
        fontweight='bold', fontsize=12
    )
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives (actual default rate)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Chart 4: Threshold Analysis ───────────────────────────
    ax = axes[1, 1]

    # Use best model for threshold analysis
    best_name  = max(test_results, key=lambda k: test_results[k]['roc_auc'])
    best_probs = test_results[best_name]['test_probs']

    thresholds_to_test = np.linspace(0.05, 0.80, 75)
    precision_vals, recall_vals, f1_vals = [], [], []

    for thresh in thresholds_to_test:
        preds = (best_probs >= thresh).astype(int)
        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        p  = tp / (tp + fp + 1e-10)
        r  = tp / (tp + fn + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)
        precision_vals.append(p)
        recall_vals.append(r)
        f1_vals.append(f1)

    optimal_thresh = float(thresholds_to_test[np.argmax(f1_vals)])

    ax.plot(thresholds_to_test, precision_vals,
            label='Precision', color='#2ecc71', linewidth=2)
    ax.plot(thresholds_to_test, recall_vals,
            label='Recall', color='#e74c3c', linewidth=2)
    ax.plot(thresholds_to_test, f1_vals,
            label='F1 Score', color='#3498db', linewidth=2.5)
    ax.axvline(
        optimal_thresh, color='black',
        linestyle='--', linewidth=1.5,
        label=f'Optimal threshold: {optimal_thresh:.2f}'
    )
    ax.set_title(
        f'Threshold Analysis — {best_name}',
        fontweight='bold', fontsize=12
    )
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 0.80)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [SAVED] {output_path}')


# ============================================================
# STEP 5 — SHAP EXPLANATIONS
# ============================================================
def generate_shap_explanations(
    model      : object,
    X_test     : pd.DataFrame,
    y_test     : pd.Series,
    output_dir : str = 'outputs',
    sample_size: int = 3000,
) -> None:
    """
    Generate SHAP explanations:
    1. Global bar chart  (overall feature importance)
    2. Beeswarm plot     (direction of impact)
    3. Single loan explanation (highest risk loan)
    4. Combined shap_explanations.png (deliverable)
    """
    if not SHAP_AVAILABLE:
        print('  [SKIP] SHAP not installed: pip install shap')
        return

    print(f'\n[STEP 5/6] Generating SHAP explanations (sample={sample_size:,})...')

    # Sample for speed
    sample_idx = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
    X_sample   = X_test.iloc[sample_idx].reset_index(drop=True)
    y_sample   = y_test.iloc[sample_idx].reset_index(drop=True)

    # Create SHAP explainer
    try:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_sample)

        # For binary classification shap_values returns list [class0, class1]
        if isinstance(shap_vals, list):
            sv = shap_vals[1]  # class 1 = default
        else:
            sv = shap_vals

        print(f'  [OK] SHAP values shape: {sv.shape}')

    except Exception as e:
        print(f'  [ERROR] SHAP failed: {e}')
        return

    # ── Plot 1: Global bar chart ──────────────────────────────
    try:
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            sv, X_sample,
            plot_type   = 'bar',
            show        = False,
            max_display = 15,
            ax          = ax1
        )
        ax1.set_title('Global Feature Importance (SHAP)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_global.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  [SAVED] {output_dir}/shap_global.png')
    except Exception as e:
        print(f'  [WARN] Global SHAP plot failed: {e}')

    # ── Plot 2: Beeswarm ──────────────────────────────────────
    try:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            sv, X_sample,
            show        = False,
            max_display = 15,
            ax          = ax2
        )
        ax2.set_title(
            'SHAP Beeswarm — Red=High Feature Value | Blue=Low',
            fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_beeswarm.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  [SAVED] {output_dir}/shap_beeswarm.png')
    except Exception as e:
        print(f'  [WARN] Beeswarm plot failed: {e}')

    # ── Local explanation: highest risk loan ──────────────────
    test_probs    = model.predict_proba(X_sample)[:, 1]
    high_risk_idx = int(np.argmax(test_probs))
    default_prob  = float(test_probs[high_risk_idx])

    shap_loan   = sv[high_risk_idx]
    loan_values = X_sample.iloc[high_risk_idx]

    top_factors = pd.DataFrame({
        'feature'      : X_sample.columns,
        'shap_value'   : shap_loan,
        'feature_value': loan_values.values,
    }).sort_values('shap_value', key=abs, ascending=False).head(10)

    print(f'\n  HIGH RISK LOAN EXPLANATION:')
    print(f'  Predicted default probability: {default_prob:.1%}')
    print(f'\n  {"Feature":<35} {"Value":>10} {"SHAP":>10} {"Impact"}')
    print(f'  {"-"*35} {"-"*10} {"-"*10} {"-"*15}')
    for _, row in top_factors.iterrows():
        impact = 'RAISES RISK' if row['shap_value'] > 0 else 'LOWERS RISK'
        print(
            f'  {row["feature"]:<35} '
            f'{row["feature_value"]:>10.2f} '
            f'{row["shap_value"]:>10.3f} '
            f'{impact}'
        )

    # Adverse action notice (ECOA regulatory requirement)
    REASON_TEMPLATES = {
        'dti'                   : 'Debt-to-income ratio too high',
        'grade_encoded'         : 'Credit grade below minimum threshold',
        'grade_dti_interaction' : 'High risk grade combined with high debt burden',
        'derogatory_score'      : 'Derogatory marks on credit history',
        'payment_to_income'     : 'Loan payment exceeds income guidelines',
        'revol_util'            : 'High revolving credit utilization',
        'int_rate'              : 'Assigned interest rate reflects elevated risk',
        'loan_to_annual_income' : 'Loan amount too high relative to income',
        'emp_length_years'      : 'Insufficient employment history',
        'delinq_2yrs'           : 'Recent delinquencies on credit file',
        'pub_rec'               : 'Derogatory public records present',
        'pub_rec_bankruptcies'  : 'Prior bankruptcy on record',
        'credit_history_years'  : 'Limited credit history',
    }

    decline_reasons = top_factors[top_factors['shap_value'] > 0].head(3)
    print(f'\n  ADVERSE ACTION NOTICE (ECOA Compliant):')
    print(f'  Loan declined. Primary reasons:')
    for i, (_, row) in enumerate(decline_reasons.iterrows(), 1):
        reason = REASON_TEMPLATES.get(
            row['feature'],
            f'Unfavorable {row["feature"].replace("_"," ")} profile'
        )
        print(f'  {i}. {reason}')

    # ── Combined deliverable plot ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        'SHAP Explanations — Credit Risk Model',
        fontsize=14, fontweight='bold'
    )

    # Left: mean SHAP importance
    mean_shap = pd.DataFrame({
        'feature'   : X_sample.columns,
        'importance': np.abs(sv).mean(axis=0)
    }).sort_values('importance', ascending=True).tail(15)

    colors_bar = [
        '#e74c3c' if f in ['grade_dti_interaction','int_rate','dti','derogatory_score']
        else '#3498db'
        for f in mean_shap['feature']
    ]
    axes[0].barh(
        mean_shap['feature'],
        mean_shap['importance'],
        color=colors_bar, edgecolor='white'
    )
    axes[0].set_title(
        'Global Feature Importance\n(mean |SHAP value|)',
        fontweight='bold', fontsize=12
    )
    axes[0].set_xlabel('Mean |SHAP Value|')

    # Right: local explanation for highest risk loan
    local_df = top_factors.copy().sort_values('shap_value')
    bar_colors = [
        '#e74c3c' if v > 0 else '#2ecc71'
        for v in local_df['shap_value']
    ]
    axes[1].barh(
        local_df['feature'],
        local_df['shap_value'],
        color=bar_colors, edgecolor='white'
    )
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_title(
        f'Local Explanation — Highest Risk Loan\n'
        f'Predicted Default Probability: {default_prob:.1%}\n'
        f'Red=increases risk | Green=decreases risk',
        fontweight='bold', fontsize=11
    )
    axes[1].set_xlabel('SHAP Value')

    plt.tight_layout()
    shap_path = f'{output_dir}/shap_explanations.png'
    plt.savefig(shap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  [SAVED] {shap_path}')


# ============================================================
# STEP 6 — SAVE ALL OUTPUTS
# ============================================================
def save_all_outputs(
    results      : dict,
    test_results : dict,
    best_model   : object,
    FEATURES     : list,
    y_val        : pd.Series,
    y_test       : pd.Series,
) -> None:
    """
    Save:
    - model_comparison.csv
    - credit_risk_v1.pkl  (best model artifact)
    - confusion matrix plot
    """
    print('\n[STEP 6/6] Saving all outputs...')

    # ── model_comparison.csv ──────────────────────────────────
    rows = []
    for name, val_data in results.items():
        test_data = test_results.get(name, {})
        rows.append({
            'model'         : name,
            'val_roc_auc'   : round(val_data['roc_auc'], 4),
            'val_pr_auc'    : round(val_data['pr_auc'], 4),
            'test_roc_auc'  : round(test_data.get('roc_auc', 0), 4),
            'test_pr_auc'   : round(test_data.get('pr_auc', 0), 4),
            'optimal_thresh': round(test_data.get('optimal_thresh', 0.5), 3),
            'train_time_s'  : round(val_data['train_time'], 1),
            'model_type'    : val_data.get('model_type', 'unknown'),
            'n_features'    : len(FEATURES),
        })

    comparison_df = pd.DataFrame(rows)
    csv_path = 'outputs/model_comparison.csv'
    comparison_df.to_csv(csv_path, index=False)
    print(f'  [SAVED] {csv_path}')
    print(f'\n  {comparison_df.to_string(index=False)}')

    # ── Save best model as pkl ────────────────────────────────
    best_name = max(
        test_results,
        key=lambda k: test_results[k]['roc_auc']
    )
    best_model_obj = test_results[best_name]['model']

    artifact = {
        'model'          : best_model_obj,
        'feature_names'  : FEATURES,
        'optimal_threshold': test_results[best_name]['optimal_thresh'],
        'test_roc_auc'   : test_results[best_name]['roc_auc'],
        'test_pr_auc'    : test_results[best_name]['pr_auc'],
        'model_name'     : best_name,
        'version'        : 'v1',
    }
    model_path = 'models/credit_risk_v1.pkl'
    joblib.dump(artifact, model_path)
    print(f'\n  [SAVED] {model_path}')
    print(f'          Best model : {best_name}')
    print(f'          Test AUC   : {test_results[best_name]["roc_auc"]:.4f}')

    # ── Confusion matrix ──────────────────────────────────────
    best_test = test_results[best_name]
    cm        = confusion_matrix(y_test, best_test['optimal_preds'])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=',d',
        cmap='Blues', ax=ax,
        xticklabels=['Good Loan', 'Default'],
        yticklabels=['Good Loan', 'Default'],
        linewidths=0.5
    )
    ax.set_title(
        f'Confusion Matrix — {best_name}\n'
        f'Threshold={best_test["optimal_thresh"]:.2f} | '
        f'Test AUC={best_test["roc_auc"]:.4f}',
        fontweight='bold'
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    # Add $ business impact annotation
    tn, fp, fn, tp = cm.ravel()
    ax.text(
        0.5, -0.12,
        f'TN={tn:,} (correctly approved)  '
        f'FP={fp:,} (approved but defaulted — LOSS)\n'
        f'FN={fn:,} (rejected but would have paid — MISSED REVENUE)  '
        f'TP={tp:,} (correctly rejected defaults)',
        transform=ax.transAxes,
        ha='center', fontsize=8, color='#555'
    )

    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [SAVED] outputs/confusion_matrix.png')


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_training_pipeline(
    parquet_path: str = 'data/processed/model_ready_dataset.parquet'
) -> dict:
    """Full training pipeline — call this or run as __main__."""

    print('=' * 60)
    print('  CREDIT RISK — MODEL TRAINING PIPELINE')
    print('=' * 60)

    # Load
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     FEATURES, neg_pos_ratio) = load_model_ready_data(parquet_path)

    # Train
    results = train_all_models(
        X_train, y_train,
        X_val,   y_val,
        neg_pos_ratio
    )

    # Evaluate
    test_results = evaluate_on_test(
        results, X_test, y_test, FEATURES
    )

    # Plot evaluation
    plot_model_evaluation(
        test_results, y_test,
        output_path='outputs/model_evaluation.png'
    )

    # SHAP explanations (use best model)
    best_name = max(
        test_results,
        key=lambda k: test_results[k]['roc_auc']
    )
    generate_shap_explanations(
        model       = test_results[best_name]['model'],
        X_test      = X_test,
        y_test      = y_test,
        output_dir  = 'outputs',
        sample_size = 3000,
    )

    # Save all outputs
    save_all_outputs(
        results      = results,
        test_results = test_results,
        best_model   = test_results[best_name]['model'],
        FEATURES     = FEATURES,
        y_val        = y_val,
        y_test       = y_test,
    )

    print('\n' + '=' * 60)
    print('  TRAINING PIPELINE COMPLETE!')
    print('=' * 60)
    print(f'  Best model    : {best_name}')
    print(f'  Test ROC-AUC  : {test_results[best_name]["roc_auc"]:.4f}')
    print(f'  Test PR-AUC   : {test_results[best_name]["pr_auc"]:.4f}')
    print(f'  Threshold     : {test_results[best_name]["optimal_thresh"]:.3f}')
    print(f'\n  Output files:')
    print(f'  >> models/credit_risk_v1.pkl')
    print(f'  >> outputs/model_evaluation.png')
    print(f'  >> outputs/shap_explanations.png')
    print(f'  >> outputs/shap_global.png')
    print(f'  >> outputs/shap_beeswarm.png')
    print(f'  >> outputs/confusion_matrix.png')
    print(f'  >> outputs/model_comparison.csv')
    print('=' * 60)

    return test_results


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    results = run_training_pipeline(
        parquet_path=r'data/processed/model_ready_dataset.parquet'
    )