import os
import sys
import logging
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# --- project paths (resolve relative to this file, works from any cwd) ---
SRC_DIR      = Path(__file__).resolve().parent
BASE_DIR     = SRC_DIR.parent
RAW_PATH     = BASE_DIR / 'data' / 'raw' / 'loan.csv'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR   = BASE_DIR / 'outputs'
MODEL_DIR    = BASE_DIR / 'models'

for _dir in [PROCESSED_DIR, OUTPUT_DIR, MODEL_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# --- column lists ---
KEEP_COLS = [
    'loan_status', 'annual_inc', 'emp_length', 'home_ownership',
    'verification_status', 'loan_amnt', 'funded_amnt', 'term',
    'int_rate', 'installment', 'grade', 'sub_grade', 'purpose',
    'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
    'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal',
    'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies',
    'issue_d', 'earliest_cr_line',
]

# columns that only exist after the loan closes — using them would be leakage
LEAKAGE_COLUMNS = [
    'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
    'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'collection_recovery_fee',
    'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d',
    'last_credit_pull_d', 'out_prncp', 'out_prncp_inv',
    'debt_settlement_flag',
]

VALID_STATUSES = {'Fully Paid', 'Charged Off', 'Default'}

CATEGORICAL_COLS = [
    'grade', 'sub_grade', 'purpose',
    'home_ownership', 'verification_status', 'loan_status',
]


def load_data(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Loading CSV: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}\n"
            f"Place loan.csv inside data/raw/ or pass --csv <path>"
        )

    # peek at headers first so we can skip columns that don't exist
    sample    = pd.read_csv(csv_path, nrows=5)
    available = [c for c in KEEP_COLS if c in sample.columns]
    skipped   = [c for c in KEEP_COLS if c not in sample.columns]

    if skipped:
        logger.warning(f"Columns not found in CSV (skipping): {skipped}")

    df = pd.read_csv(csv_path, usecols=available, low_memory=False)

    logger.info(
        f"Loaded {len(df):,} rows, {len(df.columns)} columns  "
        f"({df.memory_usage(deep=True).sum() / 1e6:.1f} MB)"
    )
    return df


def clean_string_numerics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'int_rate' in df.columns:
        if df['int_rate'].dtype == object:
            df['int_rate'] = (
                df['int_rate'].str.replace('%', '', regex=False).str.strip().astype(float)
            )
            logger.info("int_rate: stripped % sign")
        else:
            logger.info("int_rate: already numeric, skipped")

    if 'revol_util' in df.columns:
        if df['revol_util'].dtype == object:
            df['revol_util'] = pd.to_numeric(
                df['revol_util'].str.replace('%', '', regex=False).str.strip(),
                errors='coerce',
            )
            logger.info("revol_util: stripped % sign")
        else:
            logger.info("revol_util: already numeric, skipped")

    return df


def remove_leakage(df: pd.DataFrame) -> pd.DataFrame:
    found = [c for c in LEAKAGE_COLUMNS if c in df.columns]

    if found:
        df = df.drop(columns=found)
        logger.info(f"Removed {len(found)} leakage columns: {found}")
    else:
        logger.info("No leakage columns found in this dataset")

    return df


def filter_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    status_counts = df['loan_status'].value_counts()
    for status, count in status_counts.items():
        flag = 'keep' if status in VALID_STATUSES else 'drop'
        logger.info(f"  [{flag}] {status}: {count:,}")

    df = df[df['loan_status'].astype(str).isin(VALID_STATUSES)].copy()

    logger.info(
        f"Status filter: {before:,} rows -> {len(df):,} rows "
        f"({before - len(df):,} dropped)"
    )
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['target'] = (
        df['loan_status'].astype(str).isin({'Charged Off', 'Default'}).astype('int8')
    )

    rate = df['target'].mean()
    logger.info(
        f"Target created — good: {(df['target']==0).sum():,} ({1-rate:.1%})  "
        f"default: {df['target'].sum():,} ({rate:.1%})"
    )
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    missing_before = df.isnull().sum()
    missing_before = missing_before[missing_before > 0]

    if missing_before.empty:
        logger.info("No missing values found")
        return df

    logger.info(f"Missing values before imputation:\n{missing_before.to_string()}")

    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].fillna('Unknown')
        df['emp_length_missing'] = df['emp_length'].isnull().astype('int8')

    if 'revol_util' in df.columns:
        df['revol_util_missing'] = df['revol_util'].isnull().astype('int8')
        df['revol_util']         = df['revol_util'].fillna(0.0)

    if 'mort_acc' in df.columns:
        df['mort_acc'] = df['mort_acc'].fillna(0)

    if 'pub_rec_bankruptcies' in df.columns:
        median_val = df['pub_rec_bankruptcies'].median()
        df['pub_rec_bankruptcies_missing'] = df['pub_rec_bankruptcies'].isnull().astype('int8')
        df['pub_rec_bankruptcies']         = df['pub_rec_bankruptcies'].fillna(median_val)

    if 'dti' in df.columns:
        dti_median = df['dti'].median()
        df['dti']  = df['dti'].fillna(dti_median)

    if 'annual_inc' in df.columns:
        before  = len(df)
        df      = df.dropna(subset=['annual_inc'])
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"annual_inc: dropped {dropped:,} rows with missing income")

    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if not remaining.empty:
        logger.warning(f"Still missing after imputation:\n{remaining.to_string()}")
    else:
        logger.info("Zero missing values remaining")

    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mem_before = df.memory_usage(deep=True).sum() / 1e6

    # int64 -> int16 or int32 depending on value range
    for col in df.select_dtypes(include=['int64']).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= -32_768 and col_max <= 32_767:
            df[col] = df[col].astype('int16')
        elif col_min >= -2_147_483_648 and col_max <= 2_147_483_647:
            df[col] = df[col].astype('int32')

    # float64 -> float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    # object -> category for low-cardinality text columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')

    mem_after = df.memory_usage(deep=True).sum() / 1e6
    saved_pct = (1 - mem_after / mem_before) * 100
    logger.info(
        f"Memory: {mem_before:.1f} MB -> {mem_after:.1f} MB  "
        f"({saved_pct:.0f}% saved)"
    )
    return df


def parse_string_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'term' in df.columns:
        df['term'] = (
            df['term'].astype(str).str.strip()
            .str.extract(r'(\d+)')[0]
            .astype('int8')
        )
        logger.info(f"term parsed -> values: {sorted(df['term'].unique().tolist())}")

    if 'emp_length' in df.columns:
        s = df['emp_length'].astype(str).str.strip()

        emp_numeric = pd.to_numeric(s.str.extract(r'(\d+)')[0], errors='coerce')
        emp_numeric = emp_numeric.where(~s.str.contains('< 1', na=False), other=0)
        emp_numeric = emp_numeric.where(
            ~s.isin(['Unknown', 'nan', 'None', 'n/a']), other=np.nan
        )

        emp_median = emp_numeric.median()
        df['emp_length_missing'] = emp_numeric.isnull().astype('int8')
        df['emp_length_years']   = emp_numeric.fillna(emp_median)

        logger.info(
            f"emp_length_years created — "
            f"unique: {sorted(df['emp_length_years'].unique().tolist())}, "
            f"NaN filled with median ({emp_median})"
        )

    return df


def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'issue_d' in df.columns:
        df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')

    if 'earliest_cr_line' in df.columns:
        df['earliest_cr_line'] = pd.to_datetime(
            df['earliest_cr_line'], format='%b-%Y', errors='coerce'
        )

    if 'issue_d' in df.columns and 'earliest_cr_line' in df.columns:
        df['credit_history_years'] = (
            (df['issue_d'] - df['earliest_cr_line']).dt.days / 365.25
        ).clip(lower=0)
        logger.info(
            f"credit_history_years — "
            f"mean: {df['credit_history_years'].mean():.1f}, "
            f"max: {df['credit_history_years'].max():.1f}"
        )

    if 'issue_d' in df.columns:
        df['issue_year']    = df['issue_d'].dt.year.astype('Int16')
        df['issue_quarter'] = df['issue_d'].dt.quarter.astype('Int8')
        df['loan_vintage']  = (
            df['issue_year'].astype(str) + ' Q' + df['issue_quarter'].astype(str)
        )
        logger.info(
            f"issue_year range: {df['issue_year'].min()} - {df['issue_year'].max()}"
        )

    drop_dates = ['issue_d', 'earliest_cr_line']
    df = df.drop(columns=[c for c in drop_dates if c in df.columns])

    return df


def save_outputs(df: pd.DataFrame, output_dir: Path, processed_dir: Path) -> None:
    """Save cleaned Parquet, validation plots, and memory audit report."""

    # drop columns that should not go into the model
    drop_for_model = ['loan_status', 'emp_length', 'loan_vintage']
    df_model = df.drop(
        columns=[c for c in drop_for_model if c in df.columns], errors='ignore'
    )

    # save parquet
    parquet_path = processed_dir / 'cleaned_loans.parquet'
    df_model.to_parquet(parquet_path, index=False)
    logger.info(
        f"Saved {parquet_path.name} — "
        f"{df_model.shape[0]:,} rows x {df_model.shape[1]} columns  "
        f"({df_model.memory_usage(deep=True).sum() / 1e6:.1f} MB)"
    )

    # 4-panel validation chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Credit Risk — Phase 1 Validation', fontsize=14, fontweight='bold')

    if 'target' in df_model.columns:
        counts = df_model['target'].value_counts()
        bars   = axes[0, 0].bar(
            ['Fully Paid (0)', 'Default (1)'],
            counts.values,
            color=['steelblue', 'tomato'],
            edgecolor='white', width=0.5,
        )
        axes[0, 0].set_title('Target Distribution')
        axes[0, 0].set_ylabel('Number of Loans')
        for bar, val in zip(bars, counts.values):
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f'{val:,}', ha='center', fontsize=10, fontweight='bold',
            )

    if 'annual_inc' in df_model.columns:
        df_model['annual_inc'].clip(upper=250_000).hist(
            bins=60, ax=axes[0, 1], color='coral', edgecolor='white'
        )
        axes[0, 1].set_title('Annual Income (clipped at $250k)')
        axes[0, 1].set_xlabel('Annual Income ($)')
        axes[0, 1].set_ylabel('Count')

    if 'revol_util' in df_model.columns:
        df_model['revol_util'].hist(
            bins=60, ax=axes[1, 0], color='steelblue', edgecolor='white'
        )
        axes[1, 0].set_title('Revolving Utilization % (after imputation)')
        axes[1, 0].set_xlabel('Utilization %')
        axes[1, 0].set_ylabel('Count')

    if 'credit_history_years' in df_model.columns:
        df_model['credit_history_years'].hist(
            bins=60, ax=axes[1, 1], color='mediumseagreen', edgecolor='white'
        )
        axes[1, 1].set_title('Credit History Length (years)')
        axes[1, 1].set_xlabel('Years')
        axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    plot_path = output_dir / 'phase1_validation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {plot_path.name}")

    # memory audit text report
    memory_report = pd.DataFrame({
        'column'   : df_model.columns,
        'dtype'    : df_model.dtypes.values,
        'memory_MB': (df_model.memory_usage(deep=True).values[1:] / 1e6).round(5),
        'missing'  : df_model.isnull().sum().values,
    }).sort_values('memory_MB', ascending=False)

    audit_path = output_dir / 'memory_audit.txt'
    with open(audit_path, 'w', encoding='utf-8') as f:
        f.write("MEMORY AUDIT\n")
        f.write(f"Rows    : {len(df_model):,}\n")
        f.write(f"Columns : {len(df_model.columns)}\n")
        f.write(
            f"Total   : {df_model.memory_usage(deep=True).sum() / 1e6:.2f} MB\n\n"
        )
        f.write(memory_report.to_string(index=False))
    logger.info(f"Saved {audit_path.name}")

    # column summary to stdout
    logger.info("Final columns in cleaned dataset:")
    for col in sorted(df_model.columns):
        dtype   = str(df_model[col].dtype)
        missing = df_model[col].isnull().sum()
        logger.info(f"  {col:<35} {dtype:<15} missing={missing:,}")


def run_preprocessing(
    csv_path:   Path = RAW_PATH,
    output_dir: Path = OUTPUT_DIR,
    processed_dir: Path = PROCESSED_DIR,
    save_plots: bool = True,
) -> pd.DataFrame:
    """Run the full 9-step preprocessing pipeline and return the cleaned DataFrame."""

    logger.info("Starting preprocessing pipeline")

    df = load_data(csv_path)
    df = clean_string_numerics(df)
    df = remove_leakage(df)
    df = filter_loan_status(df)
    df = create_target(df)
    df = impute_missing(df)
    df = optimize_dtypes(df)
    df = parse_string_features(df)
    df = engineer_date_features(df)

    if save_plots:
        save_outputs(df, output_dir, processed_dir)

    logger.info(
        f"Pipeline complete — "
        f"{len(df):,} rows, {len(df.columns)} columns, "
        f"default rate: {df['target'].mean():.1%}"
    )
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Credit risk preprocessing pipeline')
    parser.add_argument(
        '--csv', type=Path, default=RAW_PATH,
        help='Path to raw loan CSV (default: data/raw/loan.csv)',
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip saving validation plots',
    )
    args = parser.parse_args()

    df = run_preprocessing(
        csv_path    = args.csv,
        save_plots  = not args.no_plots,
    )
