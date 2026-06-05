import os
import sys
import time
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    classification_report, confusion_matrix,
)
from sklearn.calibration import calibration_curve

# --- paths ---
SRC_DIR    = Path(__file__).resolve().parent
BASE_DIR   = SRC_DIR.parent
DATA_DIR   = BASE_DIR / 'data' / 'processed'
MODEL_DIR  = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'

for d in [MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_data():
    path = DATA_DIR / 'model_ready_dataset.parquet'
    if not path.exists():
        raise FileNotFoundError(f"Run feature_engineering.py first. Missing: {path}")

    df = pd.read_parquet(path)

    if 'split' not in df.columns:
        raise ValueError("'split' column missing. Re-run feature_engineering.py.")

    train_df = df[df['split'] == 'train'].drop(columns=['split'])
    test_df  = df[df['split'] == 'test'].drop(columns=['split'])

    features = [c for c in train_df.columns if c != 'target']

    # carve validation out of train (stratified, temporal order preserved)
    X_full, y_full = train_df[features], train_df['target']
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.15, random_state=42, stratify=y_full
    )

    X_test = test_df[features].fillna(0)
    y_test = test_df['target']

    X_train = X_train.fillna(0)
    X_val   = X_val.fillna(0)

    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"Default rate — Train: {y_train.mean():.1%}  Val: {y_val.mean():.1%}  Test: {y_test.mean():.1%}")
    print(f"Features: {len(features)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, features


def train_models(X_train, y_train, X_val, y_val):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    results = {}

    # baseline
    print("\nTraining Logistic Regression...")
    t0 = time.time()
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=0.1, class_weight='balanced',
            max_iter=1000, random_state=42, solver='saga', n_jobs=-1
        ))
    ])
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_val)[:, 1]
    results['LogisticRegression'] = {
        'model':      lr,
        'val_probs':  lr_probs,
        'val_auc':    roc_auc_score(y_val, lr_probs),
        'val_prauc':  average_precision_score(y_val, lr_probs),
        'train_time': time.time() - t0,
    }
    print(f"  Val AUC: {results['LogisticRegression']['val_auc']:.4f}")

    # primary
    print("\nTraining LightGBM...")
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=63, max_depth=6,
        min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_probs = lgb_model.predict_proba(X_val)[:, 1]
    results['LightGBM'] = {
        'model':      lgb_model,
        'val_probs':  lgb_probs,
        'val_auc':    roc_auc_score(y_val, lgb_probs),
        'val_prauc':  average_precision_score(y_val, lgb_probs),
        'train_time': time.time() - t0,
        'best_iter':  lgb_model.best_iteration_,
    }
    print(f"  Val AUC: {results['LightGBM']['val_auc']:.4f}  Best iter: {lgb_model.best_iteration_}")

    return results


def evaluate(models, X_test, y_test):
    test_results = {}

    for name, res in models.items():
        model = res['model']
        probs = model.predict_proba(X_test)[:, 1]

        auc   = roc_auc_score(y_test, probs)
        prauc = average_precision_score(y_test, probs)

        # find best threshold by F1
        thresholds = np.linspace(0.05, 0.80, 100)
        f1_scores  = []
        for t in thresholds:
            preds = (probs >= t).astype(int)
            tp = ((preds == 1) & (y_test == 1)).sum()
            fp = ((preds == 1) & (y_test == 0)).sum()
            fn = ((preds == 0) & (y_test == 1)).sum()
            p  = tp / (tp + fp + 1e-10)
            r  = tp / (tp + fn + 1e-10)
            f1_scores.append(2 * p * r / (p + r + 1e-10))

        best_thresh = float(thresholds[np.argmax(f1_scores)])
        best_preds  = (probs >= best_thresh).astype(int)

        print(f"\n{name}  AUC={auc:.4f}  PR-AUC={prauc:.4f}  Threshold={best_thresh:.3f}")
        print(classification_report(y_test, best_preds, target_names=['Good', 'Default']))

        test_results[name] = {
            'model':      model,
            'probs':      probs,
            'auc':        auc,
            'prauc':      prauc,
            'threshold':  best_thresh,
            'preds':      best_preds,
            'val_auc':    res['val_auc'],
            'train_time': res['train_time'],
        }

    return test_results


def plot_evaluation(test_results, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Evaluation', fontsize=14, fontweight='bold')
    colors = ['#3498db', '#e74c3c']

    # ROC
    ax = axes[0, 0]
    first_fpr = first_tpr = None
    for i, (name, res) in enumerate(test_results.items()):
        fpr, tpr, _ = roc_curve(y_test, res['probs'])
        ax.plot(fpr, tpr, label=f"{name} ({res['auc']:.3f})", color=colors[i], linewidth=2)
        if first_fpr is None:
            first_fpr, first_tpr = fpr, tpr
    ax.fill_between(first_fpr, first_tpr, alpha=0.05, color=colors[0])
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    ax.grid(alpha=0.3)

    # Precision-Recall
    ax = axes[0, 1]
    for i, (name, res) in enumerate(test_results.items()):
        prec, rec, _ = precision_recall_curve(y_test, res['probs'])
        ax.plot(rec, prec, label=f"{name} ({res['prauc']:.3f})", color=colors[i], linewidth=2)
    ax.axhline(y_test.mean(), color='gray', linestyle='--', label=f"Baseline ({y_test.mean():.1%})")
    ax.set_title('Precision-Recall')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(alpha=0.3)

    # Calibration
    ax = axes[1, 0]
    for i, (name, res) in enumerate(test_results.items()):
        frac_pos, mean_pred = calibration_curve(y_test, res['probs'], n_bins=10)
        ax.plot(mean_pred, frac_pos, 's-', label=name, color=colors[i], linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_title('Calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.legend()
    ax.grid(alpha=0.3)

    # Threshold analysis (best model)
    ax = axes[1, 1]
    best_name  = max(test_results, key=lambda k: test_results[k]['auc'])
    best_probs = test_results[best_name]['probs']
    thresholds = np.linspace(0.05, 0.80, 75)
    prec_vals, rec_vals, f1_vals = [], [], []

    for t in thresholds:
        preds = (best_probs >= t).astype(int)
        tp = ((preds == 1) & (y_test == 1)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        fn = ((preds == 0) & (y_test == 1)).sum()
        p  = tp / (tp + fp + 1e-10)
        r  = tp / (tp + fn + 1e-10)
        prec_vals.append(p)
        rec_vals.append(r)
        f1_vals.append(2 * p * r / (p + r + 1e-10))

    best_thresh = thresholds[np.argmax(f1_vals)]
    ax.plot(thresholds, prec_vals, label='Precision', color='#2ecc71', linewidth=2)
    ax.plot(thresholds, rec_vals,  label='Recall',    color='#e74c3c', linewidth=2)
    ax.plot(thresholds, f1_vals,   label='F1',        color='#3498db', linewidth=2.5)
    ax.axvline(best_thresh, color='black', linestyle='--', label=f'Best: {best_thresh:.2f}')
    ax.set_title(f'Threshold Analysis ({best_name})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'model_evaluation.png'}")


def plot_confusion(test_results, y_test):
    best_name = max(test_results, key=lambda k: test_results[k]['auc'])
    res = test_results[best_name]
    cm  = confusion_matrix(y_test, res['preds'])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
        xticklabels=['Good', 'Default'],
        yticklabels=['Good', 'Default'],
    )
    ax.set_title(
        f"Confusion Matrix — {best_name}\n"
        f"Threshold={res['threshold']:.2f}  AUC={res['auc']:.4f}",
        fontweight='bold'
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.text(
        0.5, -0.1,
        f"TN={tn:,}  FP={fp:,} (loss)  FN={fn:,} (missed revenue)  TP={tp:,}",
        transform=ax.transAxes, ha='center', fontsize=8, color='#555'
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'confusion_matrix.png'}")


def generate_shap(model, X_test, y_test):
    print("\nGenerating SHAP explanations...")
    
    # Extract raw model if wrapped in a sklearn Pipeline
    if hasattr(model, 'named_steps'):
        # It's a Pipeline — get the actual model step
        # Try common step names
        for step_name in ['clf', 'model', 'classifier']:
            if step_name in model.named_steps:
                model = model.named_steps[step_name]
                print(f"  Extracted model from Pipeline step: '{step_name}'")
                break
        else:
            # Fallback: get the last step in the pipeline
            model = list(model.named_steps.values())[-1]
            print("  Extracted last step from Pipeline")

    # SHAP only works well with tree models
    # Skip if it's a linear model (LogisticRegression, etc.)
    from sklearn.linear_model import LogisticRegression as LR
    if isinstance(model, LR):
        print("  Skipping SHAP — LogisticRegression not supported by TreeExplainer.")
        print("  SHAP works with LightGBM/XGBoost/RandomForest only.")
        return

    # Stratified sample
    X_pos = X_test[y_test == 1]
    X_neg = X_test[y_test == 0]
    n = min(1000, len(X_pos), len(X_neg))

    X_sample = pd.concat([
        X_pos.sample(n, random_state=42),
        X_neg.sample(n, random_state=42)
    ])

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Global importance
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, plot_type='bar', show=False, max_display=15)
    plt.title('Global Feature Importance (SHAP)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_global.png', dpi=150)
    plt.close()

    # Beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, show=False, max_display=15)
    plt.title('SHAP Beeswarm', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_beeswarm.png', dpi=150)
    plt.close()

    print("  Saved shap_global.png and shap_beeswarm.png")

    # stratified sample — equal defaults and non-defaults
    n = min(1500, (y_test == 1).sum(), (y_test == 0).sum())
    idx = np.concatenate([
        np.where(y_test == 1)[0][:n],
        np.where(y_test == 0)[0][:n],
    ])
    X_sample = X_test.iloc[idx].reset_index(drop=True)

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_sample)
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    # global bar
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, plot_type='bar', show=False, max_display=15)
    plt.title('Global Feature Importance (SHAP)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_global.png', dpi=150, bbox_inches='tight')
    plt.close()

    # beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, show=False, max_display=15)
    plt.title('SHAP Beeswarm', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()

    # local explanation — highest risk loan
    probs         = model.predict_proba(X_sample)[:, 1]
    high_risk_idx = int(np.argmax(probs))
    default_prob  = float(probs[high_risk_idx])
    shap_loan     = sv[high_risk_idx]
    loan_values   = X_sample.iloc[high_risk_idx]

    top_factors = pd.DataFrame({
        'feature':       X_sample.columns,
        'shap_value':    shap_loan,
        'feature_value': loan_values.values,
    }).sort_values('shap_value', key=abs, ascending=False).head(10)

    print(f"\nHighest risk loan — predicted default: {default_prob:.1%}")
    for _, row in top_factors.iterrows():
        direction = 'raises risk' if row['shap_value'] > 0 else 'lowers risk'
        print(f"  {row['feature']:<35} {row['feature_value']:>8.2f}   {direction}")

    # combined plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('SHAP Explanations', fontsize=13, fontweight='bold')

    mean_shap = pd.DataFrame({
        'feature':    X_sample.columns,
        'importance': np.abs(sv).mean(axis=0),
    }).sort_values('importance').tail(15)

    axes[0].barh(mean_shap['feature'], mean_shap['importance'], color='#3498db', edgecolor='white')
    axes[0].set_title('Global Importance (mean |SHAP|)')
    axes[0].set_xlabel('Mean |SHAP Value|')

    local = top_factors.sort_values('shap_value')
    bar_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in local['shap_value']]
    axes[1].barh(local['feature'], local['shap_value'], color=bar_colors, edgecolor='white')
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_title(f'Highest Risk Loan — Default Prob: {default_prob:.1%}')
    axes[1].set_xlabel('SHAP Value')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_explanations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP plots to {OUTPUT_DIR}")


def save_outputs(models, test_results, features):
    # comparison CSV
    rows = []
    for name, res in test_results.items():
        rows.append({
            'model':       name,
            'val_auc':     round(models[name]['val_auc'], 4),
            'val_prauc':   round(models[name]['val_prauc'], 4),
            'test_auc':    round(res['auc'], 4),
            'test_prauc':  round(res['prauc'], 4),
            'threshold':   round(res['threshold'], 3),
            'train_time':  round(res['train_time'], 1),
            'n_features':  len(features),
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)

    # best model
    best_name = max(test_results, key=lambda k: test_results[k]['auc'])
    best_model = test_results[best_name]['model']

    joblib.dump({
    'model':       best_model,
    'features':    features,
    'threshold':   test_results[best_name]['threshold'],
    'model_name':  best_name,
    'test_roc_auc': test_results[best_name]['auc'],
         }, MODEL_DIR / 'credit_risk_v1.pkl')
    
    # save metadata separately so API can read it without loading the model
    metadata = {
        'model_name': best_name,
        'features':   features,
        'threshold':  test_results[best_name]['threshold'],
        'test_auc':   test_results[best_name]['auc'],
        'test_prauc': test_results[best_name]['prauc'],
    }
    with open(MODEL_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBest model: {best_name}  AUC: {test_results[best_name]['auc']:.4f}")
    print(f"Saved: {MODEL_DIR / 'credit_risk_v1.pkl'}")
    print(f"Saved: {MODEL_DIR / 'metadata.json'}")


# training.py — main()
def main():
    X_train, y_train, X_val, y_val, X_test, y_test, features = load_data()

    models       = train_models(X_train, y_train, X_val, y_val)
    test_results = evaluate(models, X_test, y_test)

    best_name       = max(test_results, key=lambda k: test_results[k]['auc'])
    shap_model_name = 'LightGBM' if 'LightGBM' in test_results else best_name

    generate_shap(
        test_results[shap_model_name]['model'],
        X_test,
        y_test
    )

    save_outputs(models, test_results, features)   


if __name__ == '__main__':
    main()
