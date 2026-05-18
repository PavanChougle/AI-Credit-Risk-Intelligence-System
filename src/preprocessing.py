"""
src/preprocessing.py
====================
Phase 1 - Complete Data Cleaning Pipeline
Run this file first before anything else.

How to run:
    python src/preprocessing.py
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Fix Windows terminal encoding error
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# ============================================================
# FOLDER SETUP
# ============================================================
os.makedirs('data/processed', exist_ok=True)
os.makedirs('outputs',        exist_ok=True)
os.makedirs('models',         exist_ok=True)

print("=" * 60)
print("  CREDIT RISK - PREPROCESSING PIPELINE")
print("=" * 60)

# ============================================================
# CONSTANTS
# ============================================================

KEEP_COLS = [
    'loan_status', 'annual_inc', 'emp_length', 'home_ownership',
    'verification_status', 'loan_amnt', 'funded_amnt', 'term',
    'int_rate', 'installment', 'grade', 'sub_grade', 'purpose',
    'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
    'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal',
    'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies',
    'issue_d', 'earliest_cr_line',
]

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


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================
def load_data(csv_path: str) -> pd.DataFrame:
    """Load only needed columns from CSV."""

    print("\n[STEP 1/9] Loading CSV file...")
    print(f"  Path: {r'C:\Users\ADMIN\Desktop\Ai-Credit Risk intelligence system\data\raw\loan.csv'}")

    # Check file exists
    if not os.path.exists(r'C:\Users\ADMIN\Desktop\Ai-Credit Risk intelligence system\data\raw\loan.csv'):
        raise FileNotFoundError(
            f"\n[ERROR] File not found: {r'C:\Users\ADMIN\Desktop\Ai-Credit Risk intelligence system\data\raw\loan.csv'}"
            f"\n  Make sure your CSV is inside data/raw/ folder"
            f"\n  Current working folder: {os.getcwd()}"
        )

    # Peek at columns available in the file
    sample = pd.read_csv(r'C:\Users\ADMIN\Desktop\Ai-Credit Risk intelligence system\data\raw\loan.csv', nrows=5)

    available = [c for c in KEEP_COLS if c in sample.columns]
    skipped   = [c for c in KEEP_COLS if c not in sample.columns]

    if skipped:
        print(f"  [WARN] Columns not in your CSV (skipping):")
        for col in skipped:
            print(f"         - {col}")

    # Load full file
    df = pd.read_csv(r'C:\Users\ADMIN\Desktop\Ai-Credit Risk intelligence system\data\raw\loan.csv', usecols=available, low_memory=False)

    print(f"  [OK] Rows loaded    : {len(df):,}")
    print(f"  [OK] Columns loaded : {len(df.columns)}")
    print(f"  [OK] Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return df


# ============================================================
# STEP 2 - CLEAN STRING COLUMNS
# ============================================================
def clean_string_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns that look like '10.5%' or ' 36 months'
    into actual numbers.
    Must run BEFORE optimize_dtypes().
    """
    print("\n[STEP 2/9] Cleaning string-encoded numbers...")
    df = df.copy()

    # int_rate: '10.5%' -> 10.5
    if 'int_rate' in df.columns:
        if df['int_rate'].dtype == object:
            df['int_rate'] = (
                df['int_rate']
                .str.replace('%', '', regex=False)
                .str.strip()
                .astype(float)
            )
            print("  [OK] int_rate cleaned  ('10.5%' -> 10.5)")
        else:
            print("  [OK] int_rate already numeric - skipped")

    # revol_util: '65.3%' -> 65.3
    if 'revol_util' in df.columns:
        if df['revol_util'].dtype == object:
            df['revol_util'] = pd.to_numeric(
                df['revol_util']
                .str.replace('%', '', regex=False)
                .str.strip(),
                errors='coerce'
            )
            print("  [OK] revol_util cleaned ('65.3%' -> 65.3)")
        else:
            print("  [OK] revol_util already numeric - skipped")

    return df


# ============================================================
# STEP 3 - REMOVE LEAKAGE COLUMNS
# ============================================================
def remove_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that contain information only known
    AFTER the loan was issued.
    """
    print("\n[STEP 3/9] Removing data leakage columns...")

    found = [c for c in LEAKAGE_COLUMNS if c in df.columns]

    if found:
        df = df.drop(columns=found)
        print(f"  [OK] Removed {len(found)} leakage columns:")
        for col in found:
            print(f"       - {col}")
    else:
        print("  [OK] No leakage columns found in this dataset")

    return df


# ============================================================
# STEP 4 - FILTER LOAN STATUSES
# ============================================================
def filter_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only loans with a clear outcome:
    - 'Fully Paid'  -> good loan (target = 0)
    - 'Charged Off' -> bad loan  (target = 1)
    - 'Default'     -> bad loan  (target = 1)
    """
    print("\n[STEP 4/9] Filtering loan statuses...")

    before = len(df)

    # Show all statuses found
    status_counts = df['loan_status'].value_counts()
    print("  All statuses found in your data:")
    for status, count in status_counts.items():
        tag = "[KEEP]" if status in VALID_STATUSES else "[DROP]"
        print(f"    {tag}  {status}: {count:,}")

    # Keep only valid statuses
    df = df[df['loan_status'].astype(str).isin(VALID_STATUSES)].copy()

    print(f"\n  Rows before : {before:,}")
    print(f"  Rows after  : {len(df):,}")
    print(f"  Rows dropped: {before - len(df):,}")

    return df


# ============================================================
# STEP 5 - CREATE TARGET VARIABLE
# ============================================================
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target column:
    0 = Fully Paid  (good loan)
    1 = Default     (bad loan - Charged Off or Default)
    """
    print("\n[STEP 5/9] Creating binary target variable...")
    df = df.copy()

    df['target'] = (
        df['loan_status']
        .astype(str)
        .isin({'Charged Off', 'Default'})
        .astype('int8')
    )

    good  = (df['target'] == 0).sum()
    bad   = (df['target'] == 1).sum()
    rate  = df['target'].mean()

    print(f"  [OK] Good loans (0 = Fully Paid) : {good:,}  ({1-rate:.1%})")
    print(f"  [OK] Bad  loans (1 = Default)    : {bad:,}  ({rate:.1%})")
    print(f"  [OK] Default rate                : {rate:.1%}")

    return df


# ============================================================
# STEP 6 - FIX MISSING VALUES
# ============================================================
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using domain knowledge.

    emp_length           -> 'Unknown' (self-employed / not reported)
    revol_util           -> 0 + indicator flag
    mort_acc             -> 0 (no mortgage accounts)
    pub_rec_bankruptcies -> median + indicator flag
    dti                  -> median
    annual_inc           -> drop rows (too important to guess)
    """
    print("\n[STEP 6/9] Fixing missing values...")
    df = df.copy()

    # Report missing before
    missing_before = df.isnull().sum()
    missing_before = missing_before[missing_before > 0]

    if len(missing_before) > 0:
        print("  Missing values found:")
        for col, count in missing_before.items():
            pct = count / len(df) * 100
            print(f"    {col}: {count:,} rows missing ({pct:.1f}%)")
    else:
        print("  [OK] No missing values found!")
        return df

    # --- emp_length ---
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].fillna('Unknown')
        print("\n  [OK] emp_length          : NaN -> 'Unknown'")

    if 'emp_length' in df.columns:
        df['emp_length_missing'] = df['emp_length'].isnull().astype('int8')
        
    # --- revol_util ---
    if 'revol_util' in df.columns:
        df['revol_util_missing'] = df['revol_util'].isnull().astype('int8')
        df['revol_util']         = df['revol_util'].fillna(0.0)
        print("  [OK] revol_util          : NaN -> 0  (flag column added)")

    # --- mort_acc ---
    if 'mort_acc' in df.columns:
        df['mort_acc'] = df['mort_acc'].fillna(0)
        print("  [OK] mort_acc            : NaN -> 0")

    # --- pub_rec_bankruptcies ---
    if 'pub_rec_bankruptcies' in df.columns:
        median_val = df['pub_rec_bankruptcies'].median()
        df['pub_rec_bankruptcies_missing'] = (
            df['pub_rec_bankruptcies'].isnull().astype('int8')
        )
        df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(median_val)
        print(f"  [OK] pub_rec_bankruptcies: NaN -> {median_val} (flag added)")

    # --- dti ---
    if 'dti' in df.columns:
        dti_median = df['dti'].median()
        df['dti']  = df['dti'].fillna(dti_median)
        print(f"  [OK] dti                 : NaN -> {dti_median:.2f} (median)")

    # --- annual_inc ---
    if 'annual_inc' in df.columns:
        before = len(df)
        df     = df.dropna(subset=['annual_inc'])
        dropped = before - len(df)
        if dropped > 0:
            print(f"  [OK] annual_inc          : dropped {dropped:,} rows")

    # Report missing after
    missing_after = df.isnull().sum()
    missing_after = missing_after[missing_after > 0]

    if len(missing_after) > 0:
        print("\n  [WARN] Still missing after imputation:")
        for col, count in missing_after.items():
            print(f"    {col}: {count:,}")
    else:
        print("\n  [OK] Zero missing values remaining!")

    return df


# ============================================================
# STEP 7 - OPTIMIZE MEMORY
# ============================================================
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shrink DataFrame memory by using smaller data types.

    int64   (8 bytes) -> int16 or int32 (2-4 bytes)
    float64 (8 bytes) -> float32 (4 bytes)
    object  (big)     -> category (tiny for low-cardinality)
    """
    print("\n[STEP 7/9] Optimizing data types to save memory...")
    df = df.copy()

    mem_before = df.memory_usage(deep=True).sum() / 1e6

    # Downcast integers
    for col in df.select_dtypes(include=['int64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min >= -32_768 and col_max <= 32_767:
            df[col] = df[col].astype('int16')
        elif col_min >= -2_147_483_648 and col_max <= 2_147_483_647:
            df[col] = df[col].astype('int32')

    # Downcast floats
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    # Convert low-cardinality text -> category
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')

    mem_after = df.memory_usage(deep=True).sum() / 1e6
    saved     = (1 - mem_after / mem_before) * 100

    print(f"  Memory before : {mem_before:.1f} MB")
    print(f"  Memory after  : {mem_after:.1f} MB")
    print(f"  [OK] Saved {saved:.0f}% memory!")

    return df


# ============================================================
# STEP 8 - PARSE STRING FEATURES
# ============================================================
def parse_string_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert text features into usable numbers.

    term       : ' 36 months' -> 36
    emp_length : '10+ years'  -> 10
                 '< 1 year'   -> 0
                 'Unknown'    -> NaN
    """
    print("\n[STEP 8/9] Parsing string features...")
    df = df.copy()

    # term
    if 'term' in df.columns:
        df['term'] = (
            df['term']
            .astype(str)
            .str.strip()
            .str.extract(r'(\d+)')[0]
            .astype('int8')
        )
        print(f"  [OK] term parsed. Values: {sorted(df['term'].unique().tolist())}")

    # emp_length - vectorized (fast)
    if 'emp_length' in df.columns:
        s = df['emp_length'].astype(str).str.strip()

        emp_numeric = pd.to_numeric(
            s.str.extract(r'(\d+)')[0],
            errors='coerce'
        )

        # '< 1 year' -> 0
        emp_numeric = emp_numeric.where(
            ~s.str.contains('< 1', na=False), other=0
        )

        # 'Unknown', 'nan' -> NaN
        emp_numeric = emp_numeric.where(
            ~s.isin(['Unknown', 'nan', 'None', 'n/a']),
            other=np.nan
        )

        df['emp_length_years'] = emp_numeric
        print(f"  [OK] emp_length_years created.")
        print(f"       Unique values: {sorted(df['emp_length_years'].dropna().unique().tolist())}")
        
        emp_median = df['emp_length_years'].median()
        df['emp_length_missing'] = df['emp_length_years'].isnull().astype('int8')
        df['emp_length_years']   = df['emp_length_years'].fillna(emp_median)
        print(f"  [OK] emp_length_years: NaN filled with median ({emp_median})")
        print(f"       emp_length_missing flag added")
    return df


# ============================================================
# STEP 9 - DATE FEATURES
# ============================================================
def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw date strings into useful numeric features.

    credit_history_years : how long borrower had credit
    issue_year           : for time-based train/test split
    issue_quarter        : seasonality
    loan_vintage         : '2015 Q3' label for cohort analysis
    """
    print("\n[STEP 9/9] Engineering date features...")
    df = df.copy()

    if 'issue_d' in df.columns:
        df['issue_d'] = pd.to_datetime(
            df['issue_d'], format='%b-%Y', errors='coerce'
        )

    if 'earliest_cr_line' in df.columns:
        df['earliest_cr_line'] = pd.to_datetime(
            df['earliest_cr_line'], format='%b-%Y', errors='coerce'
        )

    # Credit history length
    if 'issue_d' in df.columns and 'earliest_cr_line' in df.columns:
        df['credit_history_years'] = (
            (df['issue_d'] - df['earliest_cr_line']).dt.days / 365.25
        ).clip(lower=0)
        print(f"  [OK] credit_history_years:"
              f" mean={df['credit_history_years'].mean():.1f},"
              f" max={df['credit_history_years'].max():.1f}")

    # Issue year and quarter
    if 'issue_d' in df.columns:
        df['issue_year']    = df['issue_d'].dt.year.astype('Int16')
        df['issue_quarter'] = df['issue_d'].dt.quarter.astype('Int8')
        df['loan_vintage']  = (
            df['issue_year'].astype(str) + ' Q' +
            df['issue_quarter'].astype(str)
        )
        print(f"  [OK] issue_year range: "
              f"{df['issue_year'].min()} to {df['issue_year'].max()}")

    # Drop raw date columns - replaced by numeric features
    drop_dates = ['issue_d', 'earliest_cr_line']
    df = df.drop(columns=[c for c in drop_dates if c in df.columns])
    print("  [OK] Raw date columns dropped (replaced by numeric features)")

    return df


# ============================================================
# SAVE OUTPUTS
# ============================================================
def save_outputs(df: pd.DataFrame) -> None:
    """Save cleaned data, plots, and memory report."""

    print("\n" + "=" * 60)
    print("  SAVING OUTPUTS")
    print("=" * 60)

    # Drop columns not needed for model
    drop_for_model = ['loan_status', 'emp_length', 'loan_vintage']
    df_model = df.drop(
        columns=[c for c in drop_for_model if c in df.columns],
        errors='ignore'
    )

    # Save parquet
    parquet_path = 'data/processed/cleaned_loans.parquet'
    df_model.to_parquet(parquet_path, index=False)
    print(f"\n  [SAVED] {parquet_path}")
    print(f"  Shape  : {df_model.shape[0]:,} rows x {df_model.shape[1]} columns")
    print(f"  Memory : {df_model.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Save validation plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Credit Risk - Phase 1 Validation',
        fontsize=14, fontweight='bold'
    )

    # Plot 1 - target distribution
    if 'target' in df_model.columns:
        counts = df_model['target'].value_counts()
        bars   = axes[0, 0].bar(
            ['Fully Paid (0)', 'Default (1)'],
            counts.values,
            color=['steelblue', 'tomato'],
            edgecolor='white', width=0.5
        )
        axes[0, 0].set_title('Target Distribution')
        axes[0, 0].set_ylabel('Number of Loans')
        for bar, val in zip(bars, counts.values):
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f'{val:,}',
                ha='center', fontsize=10, fontweight='bold'
            )

    # Plot 2 - annual income
    if 'annual_inc' in df_model.columns:
        df_model['annual_inc'].clip(upper=250_000).hist(
            bins=60, ax=axes[0, 1],
            color='coral', edgecolor='white'
        )
        axes[0, 1].set_title('Annual Income (clipped at $250k)')
        axes[0, 1].set_xlabel('Annual Income ($)')
        axes[0, 1].set_ylabel('Count')

    # Plot 3 - revolving utilization
    if 'revol_util' in df_model.columns:
        df_model['revol_util'].hist(
            bins=60, ax=axes[1, 0],
            color='steelblue', edgecolor='white'
        )
        axes[1, 0].set_title('Revolving Utilization % (after imputation)')
        axes[1, 0].set_xlabel('Utilization %')
        axes[1, 0].set_ylabel('Count')

    # Plot 4 - credit history
    if 'credit_history_years' in df_model.columns:
        df_model['credit_history_years'].hist(
            bins=60, ax=axes[1, 1],
            color='mediumseagreen', edgecolor='white'
        )
        axes[1, 1].set_title('Credit History Length (years)')
        axes[1, 1].set_xlabel('Years')
        axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    plot_path = 'outputs/phase1_validation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {plot_path}")

    # Save memory audit
    memory_report = pd.DataFrame({
        'column'   : df_model.columns,
        'dtype'    : df_model.dtypes.values,
        'memory_MB': (
            df_model.memory_usage(deep=True).values[1:] / 1e6
        ).round(5),
        'missing'  : df_model.isnull().sum().values,
    }).sort_values('memory_MB', ascending=False)

    audit_path = 'outputs/memory_audit.txt'
    with open(audit_path, 'w', encoding='utf-8') as f:
        f.write("MEMORY AUDIT REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total rows   : {len(df_model):,}\n")
        f.write(f"Total cols   : {len(df_model.columns)}\n")
        f.write(
            f"Total memory : "
            f"{df_model.memory_usage(deep=True).sum() / 1e6:.2f} MB\n"
        )
        f.write("=" * 60 + "\n\n")
        f.write(memory_report.to_string(index=False))

    print(f"  [SAVED] {audit_path}")

    # Print final column list
    print("\n  Final columns in cleaned dataset:")
    print(f"  {'Column':<35} {'Dtype':<15} {'Missing'}")
    print(f"  {'-'*35} {'-'*15} {'-'*10}")
    for col in sorted(df_model.columns):
        dtype   = str(df_model[col].dtype)
        missing = df_model[col].isnull().sum()
        print(f"  {col:<35} {dtype:<15} {missing:,}")


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_preprocessing(
    csv_path:   str  = r'C:\Users\ADMIN\Desktop\Ai-Credit Risk intelligence system\data\raw\loan.csv',
    output_dir: str  = r'C:\Users\ADMIN\Desktop\Ai-Credit Risk intelligence system\data\processed',
    save_plots: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline - call this from notebooks or other scripts.

    Args:
        csv_path   : where your raw CSV file is
        output_dir : where to save the cleaned parquet file
        save_plots : whether to generate validation charts

    Returns:
        Cleaned DataFrame ready for feature engineering
    """
    df = load_data(r'C:\Users\ADMIN\Desktop\Ai-Credit Risk intelligence system\data\raw\loan.csv')
    df = clean_string_numerics(df)
    df = remove_leakage(df)
    df = filter_loan_status(df)
    df = create_target(df)
    df = impute_missing(df)
    df = optimize_dtypes(df)
    df = parse_string_features(df)
    df = engineer_date_features(df)

    if save_plots:
        save_outputs(df)

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {len(df.columns)}")
    print(f"  Default rate : {df['target'].mean():.1%}")
    print(f"\n  Output files created:")
    print(f"  >> data/processed/cleaned_loans.parquet")
    print(f"  >> outputs/phase1_validation.png")
    print(f"  >> outputs/memory_audit.txt")
    print("=" * 60)

    return df


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':

    # --- SET YOUR CSV PATH HERE ---
    CSV_PATH = r'C:\Users\ADMIN\Desktop\Ai-Credit Risk intelligence system\data\raw\loan.csv'

    df = run_preprocessing(
        csv_path   = r'C:\Users\ADMIN\Desktop\Ai-Credit Risk intelligence system\data\raw\loan.csv',
        output_dir = r'C:\Users\ADMIN\Desktop\Ai-Credit Risk intelligence system\data\processed',
        save_plots = True,
    )