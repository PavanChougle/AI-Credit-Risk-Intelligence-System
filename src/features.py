import os
import sys
import logging
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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# --- paths ---
SRC_DIR       = Path(__file__).resolve().parent
BASE_DIR      = SRC_DIR.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR    = BASE_DIR / 'outputs'

for _dir in [PROCESSED_DIR, OUTPUT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# --- encoding maps ---

# grade: A (safest) = 1, G (riskiest) = 7
GRADE_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

# sub-grade: A1=1 through G5=35, preserves fine-grained ordering within each grade
SUBGRADE_MAP = {
    f"{g}{n}": (gi * 5) + n
    for gi, g in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    for n in range(1, 6)
}

# home ownership: higher = more stable housing situation
HOME_OWNERSHIP_MAP = {
    'RENT': 0, 'MORTGAGE': 1, 'OWN': 2,
    'ANY': 1, 'OTHER': -1, 'NONE': -1,
}

# verification: higher = more verified income
VERIFICATION_MAP = {
    'Not Verified': 0, 'Source Verified': 1, 'Verified': 2,
}


class CreditRiskEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        if 'purpose' in X.columns:
            self.purpose_categories_ = (
                X['purpose'].astype(str).value_counts().index.tolist()
            )
        else:
            self.purpose_categories_ = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'grade' in X.columns:
            X['grade_encoded'] = (
                X['grade'].astype(str).str.upper()
                .map(GRADE_MAP).fillna(4)
                .astype('int8')
            )

        if 'sub_grade' in X.columns:
            X['sub_grade_encoded'] = (
                X['sub_grade'].astype(str).str.upper()
                .map(SUBGRADE_MAP).fillna(18)
                .astype('int8')
            )

        if 'home_ownership' in X.columns:
            X['home_ownership_encoded'] = (
                X['home_ownership'].astype(str).str.upper()
                .map(HOME_OWNERSHIP_MAP).fillna(0)
                .astype('int8')
            )

        if 'verification_status' in X.columns:
            X['verification_encoded'] = (
                X['verification_status'].astype(str)
                .map(VERIFICATION_MAP).fillna(0)
                .astype('int8')
            )

        # one-hot encode loan purpose
        if 'purpose' in X.columns and self.purpose_categories_:
            for cat in self.purpose_categories_:
                col_name = 'purpose_' + cat.replace(' ', '_').replace('/', '_').lower()
                X[col_name] = (X['purpose'].astype(str) == cat).astype('int8')

        drop_cols = [
            'grade', 'sub_grade', 'home_ownership',
            'verification_status', 'purpose', 'loan_status',
        ]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')
        return X


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. monthly payment burden — underwriting red flag above 25% of income
        if 'installment' in X.columns and 'annual_inc' in X.columns:
            monthly_inc = (X['annual_inc'] / 12).clip(lower=1)
            X['payment_to_income'] = (
                X['installment'] / monthly_inc
            ).clip(upper=2).astype('float32')

        # 2. how much of annual income the loan represents
        if 'loan_amnt' in X.columns and 'annual_inc' in X.columns:
            X['loan_to_annual_income'] = (
                X['loan_amnt'] / X['annual_inc'].clip(lower=1)
            ).clip(upper=5).astype('float32')

        # 3. existing DTI + new payment burden combined
        if 'dti' in X.columns and 'payment_to_income' in X.columns:
            X['total_debt_burden'] = (
                X['dti'] + X['payment_to_income'] * 100
            ).clip(upper=200).astype('float32')

        # 4. revolving utilisation above 70% is an explicit FICO red flag
        if 'revol_util' in X.columns:
            X['high_revol_util_flag'] = (X['revol_util'] > 70).astype('int8')
            X['revol_util_squared']   = (X['revol_util'] ** 2).astype('float32')

        # 5. composite derogatory score — weights reflect severity
        #    delinquency x2, public record x3, bankruptcy x5
        delinq   = X.get('delinq_2yrs',         pd.Series(0, index=X.index))
        pub_rec  = X.get('pub_rec',              pd.Series(0, index=X.index))
        bankrupt = X.get('pub_rec_bankruptcies', pd.Series(0, index=X.index))
        X['derogatory_score'] = (
            delinq.clip(upper=5).fillna(0)   * 2 +
            pub_rec.clip(upper=3).fillna(0)  * 3 +
            bankrupt.clip(upper=2).fillna(0) * 5
        ).astype('float32')

        # 6. long credit history + many accounts = experienced borrower
        if 'credit_history_years' in X.columns and 'total_acc' in X.columns:
            X['credit_depth'] = (
                X['credit_history_years'] * X['total_acc'].clip(upper=50) / 10
            ).astype('float32')

        # 7. grade x DTI interaction — catches mispriced risk
        #    e.g. grade B borrower with DTI 40% is riskier than their grade implies
        if 'grade_encoded' in X.columns and 'dti' in X.columns:
            X['grade_dti_interaction'] = (
                X['grade_encoded'] * X['dti']
            ).astype('float32')

        # 8. DTI cliff flag — EDA showed default rates spike sharply above 30%
        if 'dti' in X.columns:
            X['dti_over_30_flag'] = (X['dti'] > 30).astype('int8')

        # 9. log transforms for right-skewed distributions
        if 'annual_inc' in X.columns:
            X['log_annual_inc'] = np.log1p(X['annual_inc']).astype('float32')

        if 'revol_bal' in X.columns:
            X['log_revol_bal'] = np.log1p(X['revol_bal']).astype('float32')

        # 10. grades E/F/G have meaningfully different default profiles
        if 'grade_encoded' in X.columns:
            X['high_risk_grade_flag'] = (X['grade_encoded'] >= 5).astype('int8')

        # 11. ratio of open to total accounts — low ratio may mean many closed/defaulted
        if 'open_acc' in X.columns and 'total_acc' in X.columns:
            X['open_acc_ratio'] = (
                X['open_acc'] / X['total_acc'].clip(lower=1)
            ).clip(0, 1).astype('float32')

        return X


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        variance_threshold:    float = 0.01,
        correlation_threshold: float = 0.85,
        top_n_features:        int   = 35,
    ):
        self.variance_threshold    = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.top_n_features        = top_n_features

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        logger.info("Fitting feature selector (3 layers)")
        self.all_input_features_ = X.columns.tolist()

        # layer 1 — variance
        vt = VarianceThreshold(threshold=self.variance_threshold)
        vt.fit(X)
        self.low_variance_features_ = X.columns[~vt.get_support()].tolist()
        X_var = X[X.columns[vt.get_support()]]
        logger.info(f"Layer 1 — low variance removed: {self.low_variance_features_}")

        # layer 2 — correlation
        corr   = X_var.corr().abs()
        upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.correlated_features_ = [
            col for col in upper.columns
            if any(upper[col] > self.correlation_threshold)
        ]
        X_corr = X_var.drop(columns=self.correlated_features_)
        logger.info(f"Layer 2 — correlated removed: {self.correlated_features_}")

        # layer 3 — model importance
        if y is not None:
            if LGBM_AVAILABLE:
                model = lgb.LGBMClassifier(
                    n_estimators=200, learning_rate=0.05,
                    num_leaves=31, random_state=42,
                    verbose=-1, n_jobs=-1,
                )
            else:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                )

            X_fit = X_corr.fillna(X_corr.median())
            model.fit(X_fit, y)

            self.importance_df_ = pd.DataFrame({
                'feature':    X_corr.columns,
                'importance': model.feature_importances_,
            }).sort_values('importance', ascending=False).reset_index(drop=True)

            self.selected_features_ = (
                self.importance_df_.head(self.top_n_features)['feature'].tolist()
            )
            logger.info(
                f"Layer 3 — top {self.top_n_features} features selected by importance"
            )
        else:
            self.selected_features_ = X_corr.columns.tolist()
            self.importance_df_ = pd.DataFrame({
                'feature':    X_corr.columns,
                'importance': [0.0] * len(X_corr.columns),
            })

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep = [f for f in self.selected_features_ if f in X.columns]
        return X[keep]

    def log_importance_table(self, top_n: int = 20):
        if not hasattr(self, 'importance_df_'):
            logger.warning("Call fit() before log_importance_table()")
            return
        logger.info(f"Top {top_n} features by importance:")
        for i, row in self.importance_df_.head(top_n).iterrows():
            logger.info(f"  {i+1:>3}.  {row['feature']:<40}  {row['importance']:.1f}")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path:   Path,
    top_n:         int = 25,
):
    top_df = importance_df.head(top_n).copy().sort_values('importance', ascending=True)

    # colour by feature family
    family_colors = {
        ('grade', 'sub_grade'):                        '#e74c3c',
        ('dti', 'debt', 'burden'):                     '#e67e22',
        ('income', 'payment', 'loan_amnt'):            '#3498db',
        ('revol', 'util'):                             '#9b59b6',
        ('delinq', 'pub_rec', 'derog'):                '#c0392b',
        ('credit_hist', 'total_acc', 'open_acc', 'depth'): '#27ae60',
        ('int_rate', 'term', 'install'):               '#f39c12',
    }

    def get_color(name):
        n = str(name).lower()
        for keys, color in family_colors.items():
            if any(k in n for k in keys):
                return color
        return '#95a5a6' if 'purpose' in n else '#7f8c8d'

    colors = [get_color(f) for f in top_df['feature']]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        'Feature Importance — Credit Risk Model',
        fontsize=15, fontweight='bold', y=0.98,
    )

    ax = fig.add_axes([0.35, 0.12, 0.60, 0.80])
    bars = ax.barh(
        range(len(top_df)), top_df['importance'],
        color=colors, edgecolor='white', height=0.7,
    )
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance Score', fontsize=11)
    ax.set_title(f'Top {top_n} Features', fontweight='bold', fontsize=12)

    max_imp = top_df['importance'].max()
    for bar, val in zip(bars, top_df['importance']):
        ax.text(
            bar.get_width() + max_imp * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.0f}', va='center', fontsize=8, color='#333',
        )

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {output_path.name}")


def save_feature_dictionary(selected_features: list, output_path: Path):
    
    families = {
        'Raw loan features':          ['loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'term'],
        'Raw borrower features':      ['annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
                                       'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
                                       'total_acc', 'mort_acc', 'pub_rec_bankruptcies',
                                       'emp_length_years'],
        'Time features':              ['credit_history_years', 'issue_year', 'issue_quarter'],
        'Encoded categoricals':       ['grade_encoded', 'sub_grade_encoded',
                                       'home_ownership_encoded', 'verification_encoded'],
        'Engineered domain features': ['payment_to_income', 'loan_to_annual_income',
                                       'total_debt_burden', 'high_revol_util_flag',
                                       'revol_util_squared', 'derogatory_score',
                                       'credit_depth', 'grade_dti_interaction',
                                       'dti_over_30_flag', 'log_annual_inc',
                                       'log_revol_bal', 'high_risk_grade_flag',
                                       'open_acc_ratio'],
        'Missing value indicators':   ['revol_util_missing', 'pub_rec_bankruptcies_missing',
                                       'emp_length_missing'],
        'Loan purpose (one-hot)':     [f for f in selected_features if f.startswith('purpose_')],
        'Target variable':            ['target'],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('# Feature dictionary — credit risk model\n\n')
        f.write(f'**Features in final model:** {len(selected_features)}\n\n')
        f.write('---\n\n')

        for family, feature_list in families.items():
            in_model = [feat for feat in feature_list if feat in selected_features]
            if not in_model:
                continue
            f.write(f'## {family}\n\n')
            for feat in in_model:
                f.write(f'- `{feat}`\n')
            f.write('\n')

        # any selected features not covered by the families above
        categorized = [feat for flist in families.values() for feat in flist]
        uncategorized = [f for f in selected_features if f not in categorized]
        if uncategorized:
            f.write('## Other selected features\n\n')
            for feat in uncategorized:
                f.write(f'- `{feat}`\n')
            f.write('\n')

        f.write('---\n')
        f.write('*Generated by src/feature_engineering.py*\n')

    logger.info(f"Saved {output_path.name}")


def run_feature_engineering(
    parquet_path:  Path = PROCESSED_DIR / 'cleaned_loans.parquet',
    output_dir:    Path = PROCESSED_DIR,
    top_n:         int  = 35,
) -> pd.DataFrame:

    logger.info("Starting feature engineering pipeline")

    # 1. load
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found. Run preprocessing.py first."
        )
    df = pd.read_parquet(parquet_path)
    logger.info(
        f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns  "
        f"({df['target'].mean():.1%} default rate)"
    )

    # 2. encode categoricals
    encoder    = CreditRiskEncoder()
    df_encoded = encoder.fit_transform(df)
    new_cols   = [c for c in df_encoded.columns if c not in df.columns]
    logger.info(
        f"Encoding — columns: {df.shape[1]} -> {df_encoded.shape[1]}  "
        f"new: {new_cols}"
    )

    # 3. engineer domain features
    engineer    = CreditFeatureEngineer()
    df_featured = engineer.fit_transform(df_encoded)
    eng_cols    = [c for c in df_featured.columns if c not in df_encoded.columns]
    logger.info(f"Engineered {len(eng_cols)} new features: {eng_cols}")
    logger.info(f"Total columns after engineering: {df_featured.shape[1]}")

    # 4. time-based train/test split (train <= 2015, test >= 2016)
    #    random split would leak future patterns into past training
    if 'issue_year' in df_featured.columns:
        train_mask = df_featured['issue_year'] <= 2015
        X_all      = df_featured.drop(columns=['target'])
        y_all      = df_featured['target']
        X_train    = X_all[train_mask]
        X_test     = X_all[~train_mask]
        y_train    = y_all[train_mask]
        y_test     = y_all[~train_mask]
        logger.info(
            f"Time-based split — train (<=2015): {len(X_train):,}  "
            f"test (2016+): {len(X_test):,}"
        )
    else:
        logger.warning("issue_year not found — falling back to random 80/20 split")
        X_all   = df_featured.drop(columns=['target'])
        y_all   = df_featured['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )

    # keep only numeric columns, fill any remaining NaN
    X_train = X_train.select_dtypes(include=[np.number]).fillna(X_train.median())
    X_test  = X_test.select_dtypes(include=[np.number]).fillna(X_train.median())

    # 5. feature selection (3-layer)
    selector = FeatureSelector(
        variance_threshold=0.01,
        correlation_threshold=0.85,
        top_n_features=top_n,
    )
    selector.fit(X_train, y_train)
    selector.log_importance_table(top_n=20)

    X_train_final = selector.transform(X_train)
    X_test_final  = selector.transform(X_test)
    logger.info(
        f"Selected {X_train_final.shape[1]} features from {X_train.shape[1]}"
    )

    # 6. save outputs
    df_train_out = X_train_final.copy()
    df_train_out['target'] = y_train.values
    df_train_out['split']  = 'train'

    df_test_out = X_test_final.copy()
    df_test_out['target'] = y_test.values
    df_test_out['split']  = 'test'

    df_model_ready = pd.concat(
        [df_train_out, df_test_out], axis=0
    ).reset_index(drop=True)

    model_path = output_dir / 'model_ready_dataset.parquet'
    df_model_ready.to_parquet(model_path, index=False)
    logger.info(
        f"Saved {model_path.name} — "
        f"{df_model_ready.shape}  "
        f"({df_model_ready.memory_usage(deep=True).sum()/1e6:.1f} MB)"
    )

    plot_feature_importance(
        importance_df=selector.importance_df_,
        output_path=OUTPUT_DIR / 'feature_importance_report.png',
        top_n=25,
    )

    save_feature_dictionary(
        selected_features=selector.selected_features_ + ['target'],
        output_path=BASE_DIR / 'feature_dictionary.md',
    )

    logger.info(
        f"Pipeline complete — "
        f"input features: {X_all.shape[1]}, "
        f"selected: {X_train_final.shape[1]}, "
        f"train rows: {len(X_train_final):,}, "
        f"test rows: {len(X_test_final):,}"
    )

    return df_model_ready


if __name__ == '__main__':
    df_model = run_feature_engineering(
        parquet_path=PROCESSED_DIR / 'cleaned_loans.parquet',
        output_dir=PROCESSED_DIR,
        top_n=35,
    )
