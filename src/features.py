"""
src/feature_engineering.py
===========================
Phase 3 - Feature Engineering Pipeline
Reusable transformer module — import this in training.py

How to run standalone:
    python src/feature_engineering.py

How to import in other files:
    from src.feature_engineering import run_feature_engineering
    df_model = run_feature_engineering('data/processed/cleaned_loans.parquet')
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
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

# LightGBM for importance ranking
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print('[WARN] lightgbm not installed. pip install lightgbm')
    print('       Falling back to RandomForest for importance.')
    from sklearn.ensemble import RandomForestClassifier

# ============================================================
# FOLDER SETUP
# ============================================================
os.makedirs('data/processed', exist_ok=True)
os.makedirs('outputs',        exist_ok=True)
os.makedirs('models',         exist_ok=True)

# ============================================================
# ENCODING MAPS
# ============================================================

# Grade: A (safest) = 1  →  G (riskiest) = 7
GRADE_MAP = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4,
    'E': 5, 'F': 6, 'G': 7
}

# Sub-grade: A1=1, A2=2, ... G5=35
# Preserves fine-grained risk ordering within each grade
SUBGRADE_MAP = {
    f"{g}{n}": (gi * 5) + n
    for gi, g in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    for n in range(1, 6)
}

# Home ownership: higher = more stable housing
HOME_OWNERSHIP_MAP = {
    'RENT'    : 0,   # least stable
    'MORTGAGE': 1,   # stable but leveraged
    'OWN'     : 2,   # most stable
    'ANY'     : 1,   # treat as mortgage
    'OTHER'   : -1,  # unknown
    'NONE'    : -1,  # unknown
}

# Verification: higher = more verified income
VERIFICATION_MAP = {
    'Not Verified'  : 0,
    'Source Verified': 1,
    'Verified'      : 2,
}


# ============================================================
# TRANSFORMER 1 — CATEGORICAL ENCODER
# ============================================================
class CreditRiskEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features for credit risk modeling.

    Encoding strategy:
    - grade        : Ordinal 1-7 (preserves risk ordering A=best, G=worst)
    - sub_grade    : Ordinal 1-35 (fine-grained within grade)
    - home_ownership: Ordinal -1 to 2 (stability hierarchy)
    - verification  : Ordinal 0-2 (verification level)
    - purpose      : One-hot (14 nominal categories, no natural order)

    Why NOT one-hot for grade?
    Because grade has a clear risk ordering — ordinal captures this.
    One-hot would lose the ordering information.

    Why one-hot for purpose?
    Because there is no natural ordering of loan purposes.
    'medical' is not 'more' or 'less' than 'vacation'.
    """

    def fit(self, X: pd.DataFrame, y=None):
        """Learn purpose categories from training data."""
        if 'purpose' in X.columns:
            self.purpose_categories_ = (
                X['purpose'].astype(str)
                .value_counts().index.tolist()
            )
        else:
            self.purpose_categories_ = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all encodings."""
        X = X.copy()

        # Grade ordinal encoding
        if 'grade' in X.columns:
            X['grade_encoded'] = (
                X['grade'].astype(str).str.upper()
                .map(GRADE_MAP).fillna(4)  # 4 = D = mid-risk default
                .astype('int8')
            )

        # Sub-grade ordinal encoding
        if 'sub_grade' in X.columns:
            X['sub_grade_encoded'] = (
                X['sub_grade'].astype(str).str.upper()
                .map(SUBGRADE_MAP).fillna(18)  # 18 = D3 = mid-risk default
                .astype('int8')
            )

        # Home ownership ordinal encoding
        if 'home_ownership' in X.columns:
            X['home_ownership_encoded'] = (
                X['home_ownership'].astype(str).str.upper()
                .map(HOME_OWNERSHIP_MAP).fillna(0)
                .astype('int8')
            )

        # Verification ordinal encoding
        if 'verification_status' in X.columns:
            X['verification_encoded'] = (
                X['verification_status'].astype(str)
                .map(VERIFICATION_MAP).fillna(0)
                .astype('int8')
            )

        # Purpose one-hot encoding
        if 'purpose' in X.columns and self.purpose_categories_:
            for cat in self.purpose_categories_:
                col_name = (
                    'purpose_' +
                    cat.replace(' ', '_').replace('/', '_').lower()
                )
                X[col_name] = (
                    X['purpose'].astype(str) == cat
                ).astype('int8')

        # Drop original categorical columns (replaced by encodings)
        drop_cols = [
            'grade', 'sub_grade', 'home_ownership',
            'verification_status', 'purpose',
            'loan_status',        # replaced by target
        ]
        X = X.drop(
            columns=[c for c in drop_cols if c in X.columns],
            errors='ignore'
        )
        return X


# ============================================================
# TRANSFORMER 2 — DOMAIN FEATURE ENGINEER
# ============================================================
class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates domain-informed features for credit risk.

    Every feature here has a BUSINESS RATIONALE —
    not just a statistical correlation.

    Features created:
    ─────────────────────────────────────────────────────────
    payment_to_income     : Monthly payment burden
    loan_to_annual_income : Loan size relative to income
    total_debt_burden     : DTI + new payment combined
    high_revol_util_flag  : Binary flag for util > 70%
    revol_util_squared    : Non-linear utilization effect
    derogatory_score      : Weighted negative credit events
    credit_depth          : History length x account breadth
    grade_dti_interaction : Grade risk x DTI pressure
    dti_over_30_flag      : Binary cliff effect at DTI=30%
    log_annual_inc        : Log transform of skewed income
    log_revol_bal         : Log transform of skewed balance
    ─────────────────────────────────────────────────────────
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # ── 1. Payment Stress Ratio ─────────────────────────
        # "What fraction of monthly income goes to this payment?"
        # Underwriting red flag: > 25% of monthly income
        if 'installment' in X.columns and 'annual_inc' in X.columns:
            monthly_inc = (X['annual_inc'] / 12).clip(lower=1)
            X['payment_to_income'] = (
                X['installment'] / monthly_inc
            ).clip(upper=2).astype('float32')

        # ── 2. Loan Burden Index ─────────────────────────────
        # "How many years of income does this loan represent?"
        # A loan that is > 50% of annual income is significant
        if 'loan_amnt' in X.columns and 'annual_inc' in X.columns:
            X['loan_to_annual_income'] = (
                X['loan_amnt'] /
                X['annual_inc'].clip(lower=1)
            ).clip(upper=5).astype('float32')

        # ── 3. Total Debt Burden ─────────────────────────────
        # "What is the borrower's TOTAL payment pressure?"
        # Existing DTI + new loan payment burden combined
        if 'dti' in X.columns and 'payment_to_income' in X.columns:
            X['total_debt_burden'] = (
                X['dti'] + X['payment_to_income'] * 100
            ).clip(upper=200).astype('float32')

        # ── 4. Credit Utilization Risk ───────────────────────
        # > 70% revolving utilization = near credit limit = stress
        # This is an explicit FICO scoring red flag threshold
        if 'revol_util' in X.columns:
            X['high_revol_util_flag'] = (
                X['revol_util'] > 70
            ).astype('int8')

            # Squared term: captures accelerating risk at high util
            X['revol_util_squared'] = (
                X['revol_util'] ** 2
            ).astype('float32')

        # ── 5. Derogatory Mark Composite ─────────────────────
        # Combines negative credit events into single risk score
        # Weights based on severity:
        #   delinquency     x2  (recent, common)
        #   public record   x3  (more serious)
        #   bankruptcy      x5  (most severe)
        delinq   = X.get('delinq_2yrs',            pd.Series(0, index=X.index))
        pub_rec  = X.get('pub_rec',                 pd.Series(0, index=X.index))
        bankrupt = X.get('pub_rec_bankruptcies',    pd.Series(0, index=X.index))

        X['derogatory_score'] = (
            delinq.clip(upper=5).fillna(0)   * 2 +
            pub_rec.clip(upper=3).fillna(0)  * 3 +
            bankrupt.clip(upper=2).fillna(0) * 5
        ).astype('float32')

        # ── 6. Credit Maturity x Depth ───────────────────────
        # Long credit history + many accounts = experienced borrower
        # A borrower with 20yr history and 30 accounts is very different
        # from someone with 20yr history and only 3 accounts
        if 'credit_history_years' in X.columns and 'total_acc' in X.columns:
            X['credit_depth'] = (
                X['credit_history_years'] *
                X['total_acc'].clip(upper=50) / 10
            ).astype('float32')

        # ── 7. Grade x DTI Interaction ───────────────────────
        # Captures mispriced risk:
        # A Grade B borrower with DTI=40% may be riskier than priced
        if 'grade_encoded' in X.columns and 'dti' in X.columns:
            X['grade_dti_interaction'] = (
                X['grade_encoded'] * X['dti']
            ).astype('float32')

        # ── 8. DTI Cliff Flag ────────────────────────────────
        # From EDA: default rates spike sharply above DTI=30%
        # Binary flag captures this non-linear relationship
        if 'dti' in X.columns:
            X['dti_over_30_flag'] = (
                X['dti'] > 30
            ).astype('int8')

        # ── 9. Log Transforms ────────────────────────────────
        # annual_inc and revol_bal are heavily right-skewed
        # Log transform makes distributions more normal
        # log1p(x) = log(x+1) — safe for zero values
        if 'annual_inc' in X.columns:
            X['log_annual_inc'] = (
                np.log1p(X['annual_inc'])
            ).astype('float32')

        if 'revol_bal' in X.columns:
            X['log_revol_bal'] = (
                np.log1p(X['revol_bal'])
            ).astype('float32')

        # ── 10. High Risk Grade Flag ─────────────────────────
        # Grades E, F, G have meaningfully different default profiles
        if 'grade_encoded' in X.columns:
            X['high_risk_grade_flag'] = (
                X['grade_encoded'] >= 5
            ).astype('int8')

        # ── 11. Open Account Ratio ───────────────────────────
        # Open accounts / total accounts = account health ratio
        # Very low ratio may indicate many closed/defaulted accounts
        if 'open_acc' in X.columns and 'total_acc' in X.columns:
            total_acc_safe = X['total_acc'].clip(lower=1)
            X['open_acc_ratio'] = (
                X['open_acc'] / total_acc_safe
            ).clip(0, 1).astype('float32')

        return X


# ============================================================
# TRANSFORMER 3 — FEATURE SELECTOR
# ============================================================
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Three-layer feature selection:
    Layer 1 — Variance filter  (remove near-constant features)
    Layer 2 — Correlation filter (remove redundant features)
    Layer 3 — Model-based importance (keep top N features)

    Why three layers?
    - Single filter methods miss redundancy
    - Model importance alone keeps correlated pairs
    - Three layers progressively remove noise
    """

    def __init__(
        self,
        variance_threshold : float = 0.01,
        correlation_threshold: float = 0.85,
        top_n_features     : int   = 35,
    ):
        self.variance_threshold    = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.top_n_features        = top_n_features

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Learn which features to keep from training data."""
        print('\n  [SELECTOR] Fitting feature selector...')
        self.all_input_features_ = X.columns.tolist()

        # ── Layer 1: Variance Filter ─────────────────────────
        vt = VarianceThreshold(threshold=self.variance_threshold)
        vt.fit(X)
        self.low_variance_features_ = (
            X.columns[~vt.get_support()].tolist()
        )
        X_var = X[X.columns[vt.get_support()]]
        print(f'  [L1] Low variance removed: {self.low_variance_features_}')

        # ── Layer 2: Correlation Filter ──────────────────────
        corr_matrix   = X_var.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self.correlated_features_ = [
            col for col in upper_triangle.columns
            if any(upper_triangle[col] > self.correlation_threshold)
        ]
        X_corr = X_var.drop(columns=self.correlated_features_)
        print(f'  [L2] Correlated removed   : {self.correlated_features_}')

        # ── Layer 3: Model-Based Importance ──────────────────
        if y is not None:
            if LGBM_AVAILABLE:
                model = lgb.LGBMClassifier(
                    n_estimators  = 200,
                    learning_rate = 0.05,
                    num_leaves    = 31,
                    random_state  = 42,
                    verbose       = -1,
                    n_jobs        = -1,
                )
            else:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators = 100,
                    random_state = 42,
                    n_jobs       = -1,
                )

            # Fill NaN for model fitting
            X_fit = X_corr.fillna(X_corr.median())
            model.fit(X_fit, y)

            self.importance_df_ = pd.DataFrame({
                'feature'   : X_corr.columns,
                'importance': model.feature_importances_,
            }).sort_values('importance', ascending=False).reset_index(drop=True)

            self.selected_features_ = (
                self.importance_df_
                .head(self.top_n_features)['feature']
                .tolist()
            )
            print(f'  [L3] Top {self.top_n_features} features selected by importance')
        else:
            # No target provided — keep all post-correlation features
            self.selected_features_ = X_corr.columns.tolist()
            self.importance_df_     = pd.DataFrame({
                'feature'   : X_corr.columns,
                'importance': [0.0] * len(X_corr.columns),
            })

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection."""
        keep = [f for f in self.selected_features_ if f in X.columns]
        return X[keep]

    def print_importance_table(self, top_n: int = 20):
        """Print top features with importance scores."""
        if not hasattr(self, 'importance_df_'):
            print('[WARN] Call fit() first')
            return
        print(f'\n  TOP {top_n} FEATURES BY MODEL IMPORTANCE:')
        print(f'  {"Rank":<6} {"Feature":<40} {"Importance":>12}')
        print(f'  {"-"*6} {"-"*40} {"-"*12}')
        for i, row in self.importance_df_.head(top_n).iterrows():
            print(
                f'  {i+1:<6} {row["feature"]:<40} '
                f'{row["importance"]:>12.1f}'
            )


# ============================================================
# VISUALIZATION — Feature Importance Report
# ============================================================
def plot_feature_importance(
    importance_df : pd.DataFrame,
    output_path   : str = 'outputs/feature_importance_report.png',
    top_n         : int = 25,
):
    """
    Generate feature importance report chart.
    Saved to outputs/feature_importance_report.png
    """
    print(f'\n  [PLOT] Generating feature importance report...')

    top_df = importance_df.head(top_n).copy()
    top_df = top_df.sort_values('importance', ascending=True)

    # Color code by feature family
    def get_color(feature_name):
        f = str(feature_name).lower()
        if any(x in f for x in ['grade', 'sub_grade']):
            return '#e74c3c'       # red = grade features
        elif any(x in f for x in ['dti', 'debt', 'burden']):
            return '#e67e22'       # orange = debt features
        elif any(x in f for x in ['income', 'payment', 'loan_amnt']):
            return '#3498db'       # blue = income/loan features
        elif any(x in f for x in ['revol', 'util']):
            return '#9b59b6'       # purple = utilization
        elif any(x in f for x in ['delinq', 'pub_rec', 'derog']):
            return '#c0392b'       # dark red = derogatory
        elif any(x in f for x in ['credit_hist', 'total_acc', 'open_acc', 'depth']):
            return '#27ae60'       # green = credit history
        elif 'purpose' in f:
            return '#95a5a6'       # grey = purpose flags
        elif any(x in f for x in ['int_rate', 'term', 'install']):
            return '#f39c12'       # yellow = loan terms
        else:
            return '#7f8c8d'       # default grey

    colors = [get_color(f) for f in top_df['feature']]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        'Feature Importance Report — Credit Risk Model',
        fontsize=15, fontweight='bold', y=0.98
    )

    # ── Main importance bar chart ──────────────────────────
    ax_main = fig.add_axes([0.35, 0.12, 0.60, 0.80])

    bars = ax_main.barh(
        range(len(top_df)),
        top_df['importance'],
        color=colors, edgecolor='white', height=0.7
    )
    ax_main.set_yticks(range(len(top_df)))
    ax_main.set_yticklabels(top_df['feature'], fontsize=10)
    ax_main.set_xlabel('Feature Importance Score', fontsize=11)
    ax_main.set_title(
        f'Top {top_n} Features by Model Importance',
        fontweight='bold', fontsize=12
    )

    # Add value labels
    max_imp = top_df['importance'].max()
    for bar, val in zip(bars, top_df['importance']):
        ax_main.text(
            bar.get_width() + max_imp * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.0f}',
            va='center', fontsize=8, color='#333'
        )

    # ── Legend for feature families ───────────────────────
    legend_items = {
        'Grade Features'       : '#e74c3c',
        'Debt / DTI Features'  : '#e67e22',
        'Income / Loan Size'   : '#3498db',
        'Revolving Utilization': '#9b59b6',
        'Derogatory Events'    : '#c0392b',
        'Credit History'       : '#27ae60',
        'Loan Purpose (OHE)'   : '#95a5a6',
        'Loan Terms'           : '#f39c12',
        'Other'                : '#7f8c8d',
    }
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=label)
        for label, c in legend_items.items()
    ]
    ax_main.legend(
        handles=legend_patches,
        loc='lower right', fontsize=8,
        title='Feature Family', title_fontsize=9
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [SAVED] {output_path}')


# ============================================================
# SAVE FEATURE DICTIONARY
# ============================================================
def save_feature_dictionary(
    selected_features: list,
    output_path: str = 'feature_dictionary.md'
):
    """
    Write business meaning of every feature to markdown file.
    This is the deliverable: feature_dictionary.md
    """

    # Business definitions for every possible feature
    FEATURE_DEFINITIONS = {
        # ── Raw numeric features ──────────────────────────────
        'loan_amnt'              : ('Loan Size', 'Total amount requested by borrower in USD.'),
        'funded_amnt'            : ('Funded Amount', 'Amount actually funded by investors. Usually equals loan_amnt.'),
        'int_rate'               : ('Interest Rate', 'Annual interest rate on the loan (%). Higher = riskier borrower.'),
        'installment'            : ('Monthly Payment', 'Fixed monthly payment amount in USD.'),
        'annual_inc'             : ('Annual Income', 'Self-reported annual income in USD. Right-skewed — use log transform.'),
        'dti'                    : ('Debt-to-Income Ratio', 'Monthly debt payments / monthly gross income (%). Key risk signal.'),
        'delinq_2yrs'            : ('Delinquencies (2yr)', 'Number of 30+ day delinquencies in past 2 years.'),
        'inq_last_6mths'         : ('Credit Inquiries (6mo)', 'Number of hard credit pulls in last 6 months. High count = credit-seeking behavior.'),
        'open_acc'               : ('Open Accounts', 'Number of currently open credit lines.'),
        'pub_rec'                : ('Public Records', 'Number of derogatory public records (liens, judgments).'),
        'revol_bal'              : ('Revolving Balance', 'Total outstanding revolving credit balance in USD.'),
        'revol_util'             : ('Revolving Utilization', 'Revolving credit used / revolving credit limit (%). >70% is a red flag.'),
        'total_acc'              : ('Total Accounts', 'Total number of credit lines ever opened.'),
        'mort_acc'               : ('Mortgage Accounts', 'Number of mortgage accounts. Homeowners tend to be more stable.'),
        'pub_rec_bankruptcies'   : ('Bankruptcies', 'Number of public record bankruptcies. Severe derogatory event.'),
        'term'                   : ('Loan Term', 'Loan duration in months. Either 36 or 60.'),

        # ── Engineered: Time ──────────────────────────────────
        'credit_history_years'   : ('Credit History Length', 'Years between earliest credit line and loan issue date. Longer = more experienced borrower = lower risk.'),
        'issue_year'             : ('Issue Year', 'Year loan was issued. Used for vintage analysis and temporal train/test split.'),
        'issue_quarter'          : ('Issue Quarter', 'Quarter loan was issued (1-4). Captures seasonal lending patterns.'),

        # ── Engineered: Encoding ─────────────────────────────
        'grade_encoded'          : ('Grade (Encoded)', 'Loan grade A-G encoded as 1-7. Higher = higher risk. Ordinal encoding preserves ordering.'),
        'sub_grade_encoded'      : ('Sub-Grade (Encoded)', 'Fine-grained grade A1-G5 encoded as 1-35. More precise risk tier.'),
        'home_ownership_encoded' : ('Home Ownership (Encoded)', 'Housing stability: OWN=2, MORTGAGE=1, RENT=0, OTHER=-1.'),
        'verification_encoded'   : ('Verification Level (Encoded)', 'Income verification: Verified=2, Source Verified=1, Not Verified=0.'),

        # ── Engineered: Domain Features ───────────────────────
        'payment_to_income'      : ('Payment-to-Income Ratio', 'Monthly loan payment / monthly income. >25% is underwriting red flag. Direct measure of new payment burden.'),
        'loan_to_annual_income'  : ('Loan-to-Income Ratio', 'Loan amount / annual income. Measures loan size relative to earnings capacity.'),
        'total_debt_burden'      : ('Total Debt Burden', 'Existing DTI + new payment burden combined. Captures full payment pressure.'),
        'high_revol_util_flag'   : ('High Utilization Flag', 'Binary: 1 if revolving utilization > 70%. Explicit FICO scoring red flag threshold.'),
        'revol_util_squared'     : ('Utilization Squared', 'revol_util squared. Captures accelerating risk at high utilization (non-linear effect).'),
        'derogatory_score'       : ('Derogatory Score', 'Weighted composite of negative credit events. delinq*2 + pub_rec*3 + bankruptcy*5. Higher = worse credit behavior.'),
        'credit_depth'           : ('Credit Depth', 'credit_history_years x total_acc / 10. Experienced + broad credit profile = lower risk.'),
        'grade_dti_interaction'  : ('Grade x DTI Interaction', 'grade_encoded * dti. Captures mispriced risk: a Grade B borrower with high DTI is riskier than priced.'),
        'dti_over_30_flag'       : ('DTI Over 30 Flag', 'Binary: 1 if DTI > 30%. From EDA: default rates spike sharply above this threshold (cliff effect).'),
        'log_annual_inc'         : ('Log Annual Income', 'log1p(annual_inc). Log transform corrects right-skew. More useful for linear models.'),
        'log_revol_bal'          : ('Log Revolving Balance', 'log1p(revol_bal). Log transform corrects right-skew of balance distribution.'),
        'high_risk_grade_flag'   : ('High Risk Grade Flag', 'Binary: 1 if grade is E, F, or G. These grades have meaningfully higher default profiles.'),
        'open_acc_ratio'         : ('Open Account Ratio', 'open_acc / total_acc. Low ratio may indicate many closed or defaulted accounts.'),

        # ── Missing Indicators ────────────────────────────────
        'revol_util_missing'          : ('Revol Util Missing Flag', 'Binary: 1 if revol_util was originally missing. May signal no revolving credit.'),
        'pub_rec_bankruptcies_missing': ('Bankruptcy Missing Flag', 'Binary: 1 if pub_rec_bankruptcies was originally missing.'),
        'emp_length_missing'          : ('Employment Missing Flag', 'Binary: 1 if emp_length was not reported. Self-employed or gig workers often omit this.'),
        'emp_length_years'            : ('Employment Length', 'Years at current employer (0-10+). Parsed from text. Longer = more stable income.'),

        # ── Purpose One-Hot ───────────────────────────────────
        'purpose_debt_consolidation'  : ('Purpose: Debt Consolidation', 'Largest segment. Near-average default rate. Borrowers combining existing debts.'),
        'purpose_credit_card'         : ('Purpose: Credit Card', 'Below-average default rate. Financially disciplined borrowers paying off cards.'),
        'purpose_home_improvement'    : ('Purpose: Home Improvement', 'Below-average default rate. Homeowners investing in property.'),
        'purpose_small_business'      : ('Purpose: Small Business', 'HIGHEST default rate (30-40%). High variance — hardest segment to model.'),
        'purpose_other'               : ('Purpose: Other', 'Catch-all category. Average default rate.'),
        'purpose_medical'             : ('Purpose: Medical', 'Above-average default. Unexpected hardship / income shock.'),
        'purpose_major_purchase'      : ('Purpose: Major Purchase', 'Near-average default rate.'),
        'purpose_car'                 : ('Purpose: Car', 'Below-average default. Secured by asset.'),
        'purpose_vacation'            : ('Purpose: Vacation', 'Above-average default. Discretionary / impulsive borrowing.'),
        'purpose_moving'              : ('Purpose: Moving', 'Near-average default rate.'),
        'purpose_wedding'             : ('Purpose: Wedding', 'Above-average default. One-time discretionary event.'),
        'purpose_house'               : ('Purpose: House', 'Near-average default rate.'),
        'purpose_educational'         : ('Purpose: Educational', 'Near-average default rate.'),
        'purpose_renewable_energy'    : ('Purpose: Renewable Energy', 'Small volume. Near-average default.'),

        # ── Target ────────────────────────────────────────────
        'target'                 : ('Target (Default)', 'Binary outcome: 1=Default/Charged Off, 0=Fully Paid. This is what we predict.'),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('# Feature Dictionary — Credit Risk Intelligence\n\n')
        f.write('Business meaning of every feature in the model.\n\n')
        f.write(f'**Total features in final model:** {len(selected_features)}\n\n')
        f.write('---\n\n')

        # Group features by family
        families = {
            'Raw Loan Features'         : ['loan_amnt','funded_amnt','int_rate','installment','term'],
            'Raw Borrower Features'     : ['annual_inc','dti','delinq_2yrs','inq_last_6mths',
                                           'open_acc','pub_rec','revol_bal','revol_util',
                                           'total_acc','mort_acc','pub_rec_bankruptcies',
                                           'emp_length_years'],
            'Time Features'             : ['credit_history_years','issue_year','issue_quarter'],
            'Encoded Categorical'       : ['grade_encoded','sub_grade_encoded',
                                           'home_ownership_encoded','verification_encoded'],
            'Engineered Domain Features': ['payment_to_income','loan_to_annual_income',
                                           'total_debt_burden','high_revol_util_flag',
                                           'revol_util_squared','derogatory_score',
                                           'credit_depth','grade_dti_interaction',
                                           'dti_over_30_flag','log_annual_inc',
                                           'log_revol_bal','high_risk_grade_flag',
                                           'open_acc_ratio'],
            'Missing Value Indicators'  : ['revol_util_missing','pub_rec_bankruptcies_missing',
                                           'emp_length_missing'],
            'Loan Purpose (One-Hot)'    : [f for f in selected_features if f.startswith('purpose_')],
            'Target Variable'           : ['target'],
        }

        for family, feature_list in families.items():
            # Only show features that are in our selected set
            family_features = [
                f for f in feature_list if f in selected_features
            ]
            if not family_features:
                continue

            f.write(f'## {family}\n\n')
            f.write('| Feature | Business Name | Description |\n')
            f.write('|---------|---------------|-------------|\n')

            for feat in family_features:
                if feat in FEATURE_DEFINITIONS:
                    biz_name, desc = FEATURE_DEFINITIONS[feat]
                else:
                    biz_name = feat.replace('_', ' ').title()
                    desc     = 'Auto-generated feature.'
                f.write(f'| `{feat}` | {biz_name} | {desc} |\n')

            f.write('\n')

        # Features in model but not categorized above
        categorized = [f for flist in families.values() for f in flist]
        uncategorized = [f for f in selected_features if f not in categorized]
        if uncategorized:
            f.write('## Other Selected Features\n\n')
            f.write('| Feature | Description |\n')
            f.write('|---------|-------------|\n')
            for feat in uncategorized:
                if feat in FEATURE_DEFINITIONS:
                    _, desc = FEATURE_DEFINITIONS[feat]
                else:
                    desc = 'Selected by model importance.'
                f.write(f'| `{feat}` | {desc} |\n')
            f.write('\n')

        f.write('---\n')
        f.write('*Generated automatically by src/feature_engineering.py*\n')

    print(f'  [SAVED] {output_path}')


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_feature_engineering(
    parquet_path : str = 'data/processed/cleaned_loans.parquet',
    output_dir   : str = 'data/processed',
    top_n        : int = 35,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps:
        1. Load cleaned data from preprocessing
        2. Encode categorical features
        3. Engineer domain features
        4. Train/test split (time-based)
        5. Three-layer feature selection
        6. Save outputs

    Args:
        parquet_path : cleaned_loans.parquet from preprocessing
        output_dir   : where to save model_ready_dataset.parquet
        top_n        : number of features to keep

    Returns:
        Model-ready DataFrame with selected features + target
    """
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('  FEATURE ENGINEERING PIPELINE')
    print('=' * 60)

    # ── 1. Load ───────────────────────────────────────────────
    print(f'\n[1/6] Loading cleaned data...')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f'[ERROR] {parquet_path} not found.\n'
            f'Run preprocessing first: python src/preprocessing.py'
        )
    df = pd.read_parquet(parquet_path)
    print(f'  [OK] Shape : {df.shape[0]:,} rows x {df.shape[1]} cols')
    print(f'  [OK] Target: {df["target"].mean():.1%} default rate')

    # ── 2. Encode Categoricals ────────────────────────────────
    print(f'\n[2/6] Encoding categorical features...')
    encoder    = CreditRiskEncoder()
    df_encoded = encoder.fit_transform(df)
    new_cols   = [c for c in df_encoded.columns if c not in df.columns]
    print(f'  [OK] Columns before: {df.shape[1]}')
    print(f'  [OK] Columns after : {df_encoded.shape[1]}')
    print(f'  [OK] New columns   : {new_cols}')

    # ── 3. Engineer Domain Features ───────────────────────────
    print(f'\n[3/6] Engineering domain features...')
    feature_eng  = CreditFeatureEngineer()
    df_featured  = feature_eng.fit_transform(df_encoded)
    eng_cols     = [c for c in df_featured.columns if c not in df_encoded.columns]
    print(f'  [OK] Engineered features added:')
    for c in eng_cols:
        print(f'       + {c}')
    print(f'  [OK] Total columns: {df_featured.shape[1]}')

    # ── 4. Train/Test Split ───────────────────────────────────
    print(f'\n[4/6] Creating train/test split...')

    # TIME-BASED SPLIT — prevents data leakage from future to past
    # Train: 2007-2015 | Test: 2016+
    if 'issue_year' in df_featured.columns:
        train_mask = df_featured['issue_year'] <= 2015
        X_all      = df_featured.drop(columns=['target'])
        y_all      = df_featured['target']
        X_train    = X_all[train_mask]
        X_test     = X_all[~train_mask]
        y_train    = y_all[train_mask]
        y_test     = y_all[~train_mask]
        print(f'  [OK] Time-based split:')
        print(f'       Train (2007-2015): {len(X_train):,} rows')
        print(f'       Test  (2016+)    : {len(X_test):,} rows')
    else:
        # Fallback to random split
        X_all   = df_featured.drop(columns=['target'])
        y_all   = df_featured['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2,
            random_state=42, stratify=y_all
        )
        print(f'  [OK] Random split (no issue_year found):')
        print(f'       Train: {len(X_train):,} | Test: {len(X_test):,}')

    # Drop non-numeric columns before selection
    X_train = X_train.select_dtypes(include=[np.number])
    X_test  = X_test.select_dtypes(include=[np.number])

    # Fill any remaining NaN
    X_train = X_train.fillna(X_train.median())
    X_test  = X_test.fillna(X_train.median())  # use train median for test

    # ── 5. Feature Selection ──────────────────────────────────
    print(f'\n[5/6] Running three-layer feature selection...')
    selector = FeatureSelector(
        variance_threshold    = 0.01,
        correlation_threshold = 0.85,
        top_n_features        = top_n,
    )
    selector.fit(X_train, y_train)
    selector.print_importance_table(top_n=20)

    X_train_final = selector.transform(X_train)
    X_test_final  = selector.transform(X_test)

    print(f'\n  [OK] Final feature count: {X_train_final.shape[1]}')
    print(f'  [OK] Final features:')
    for f in selector.selected_features_:
        print(f'       - {f}')

    # ── 6. Save Outputs ───────────────────────────────────────
    print(f'\n[6/6] Saving outputs...')
    print('=' * 60)

    # Save model-ready dataset (train + test combined with target)
    df_train_out = X_train_final.copy()
    df_train_out['target'] = y_train.values
    df_train_out['split']  = 'train'

    df_test_out = X_test_final.copy()
    df_test_out['target'] = y_test.values
    df_test_out['split']  = 'test'

    df_model_ready = pd.concat(
        [df_train_out, df_test_out], axis=0
    ).reset_index(drop=True)

    model_path = os.path.join(output_dir, 'model_ready_dataset.parquet')
    df_model_ready.to_parquet(model_path, index=False)
    print(f'\n  [SAVED] {model_path}')
    print(f'          Shape  : {df_model_ready.shape}')
    print(f'          Memory : {df_model_ready.memory_usage(deep=True).sum()/1e6:.1f} MB')

    # Save feature importance chart
    plot_feature_importance(
        importance_df = selector.importance_df_,
        output_path   = 'outputs/feature_importance_report.png',
        top_n         = 25,
    )

    # Save feature dictionary
    all_features = selector.selected_features_ + ['target']
    save_feature_dictionary(
        selected_features = all_features,
        output_path       = 'feature_dictionary.md',
    )

    print('\n' + '=' * 60)
    print('  FEATURE ENGINEERING COMPLETE!')
    print('=' * 60)
    print(f'  Input features  : {X_all.shape[1]}')
    print(f'  Selected features: {X_train_final.shape[1]}')
    print(f'  Train rows      : {len(X_train_final):,}')
    print(f'  Test  rows      : {len(X_test_final):,}')
    print(f'\n  Output files:')
    print(f'  >> data/processed/model_ready_dataset.parquet')
    print(f'  >> outputs/feature_importance_report.png')
    print(f'  >> feature_dictionary.md')
    print('=' * 60)

    return df_model_ready


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    df_model = run_feature_engineering(
        parquet_path = r'data/processed/cleaned_loans.parquet',
        output_dir   = r'data/processed',
        top_n        = 35,
    )