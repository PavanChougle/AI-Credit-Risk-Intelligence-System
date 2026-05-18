"""
api/app.py
==========
FastAPI Credit Risk Scoring Endpoint

How to run:
    uvicorn api.app:app --reload --port 8000

Test endpoints:
    http://localhost:8000/health
    http://localhost:8000/docs   (interactive UI)
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ============================================================
# LOAD MODEL AT STARTUP
# ============================================================
MODEL_PATH = 'models/credit_risk_v1.pkl'

try:
    artifact        = joblib.load(MODEL_PATH)
    MODEL           = artifact['model']
    FEATURE_NAMES   = artifact['feature_names']
    OPT_THRESHOLD   = artifact['optimal_threshold']
    MODEL_NAME      = artifact['model_name']
    MODEL_AUC       = artifact['test_roc_auc']
    print(f'[OK] Model loaded: {MODEL_NAME}  AUC={MODEL_AUC:.4f}')
except FileNotFoundError:
    MODEL         = None
    FEATURE_NAMES = []
    OPT_THRESHOLD = 0.5
    MODEL_NAME    = 'not_loaded'
    MODEL_AUC     = 0.0
    print(f'[WARN] Model not found at {MODEL_PATH}')
    print('       Run python src/training.py first')

# ============================================================
# RISK TIER DEFINITIONS
# ============================================================
RISK_TIERS = {
    'ACCEPT': {
        'threshold'        : (0.00, 0.12),
        'label'            : 'Accept - Standard Terms',
        'action'           : 'Auto-approve. Standard underwriting applies.',
        'expected_loss_pct': 'Less than 3 percent portfolio loss',
        'color'            : 'green',
    },
    'REVIEW': {
        'threshold'        : (0.12, 0.25),
        'label'            : 'Manual Review Required',
        'action'           : 'Request income documentation. Human review.',
        'expected_loss_pct': '3 to 8 percent portfolio loss',
        'color'            : 'yellow',
    },
    'CAUTION': {
        'threshold'        : (0.25, 0.40),
        'label'            : 'High Risk - Conditional Offer',
        'action'           : 'Offer reduced amount or require co-signer.',
        'expected_loss_pct': '8 to 15 percent portfolio loss',
        'color'            : 'orange',
    },
    'DECLINE': {
        'threshold'        : (0.40, 1.01),
        'label'            : 'Decline - Risk Too High',
        'action'           : 'Issue adverse action notice with reasons.',
        'expected_loss_pct': 'Greater than 15 percent portfolio loss',
        'color'            : 'red',
    },
}

ADVERSE_ACTION_REASONS = {
    'int_rate'              : 'Interest rate reflects elevated credit risk profile',
    'dti'                   : 'Debt-to-income ratio exceeds acceptable threshold',
    'grade_dti_interaction' : 'Credit grade combined with debt burden is too high',
    'derogatory_score'      : 'Derogatory marks present on credit history',
    'revol_util'            : 'Revolving credit utilization is too high',
    'loan_to_annual_income' : 'Requested loan amount too high relative to income',
    'emp_length_years'      : 'Insufficient verified employment history',
    'delinq_2yrs'           : 'Recent delinquencies present on credit file',
    'pub_rec'               : 'Derogatory public records on file',
    'pub_rec_bankruptcies'  : 'Prior bankruptcy on record',
    'credit_history_years'  : 'Insufficient length of credit history',
    'high_risk_grade_flag'  : 'Credit grade below minimum required threshold',
    'credit_depth'          : 'Limited breadth of credit experience',
}


def get_risk_tier(probability: float) -> dict:
    """Map predicted default probability to business risk tier."""
    for tier_name, info in RISK_TIERS.items():
        lo, hi = info['threshold']
        if lo <= probability < hi:
            return {
                'tier'         : tier_name,
                'label'        : info['label'],
                'action'       : info['action'],
                'expected_loss': info['expected_loss_pct'],
                'color'        : info['color'],
            }
    return get_risk_tier(0.99)


# ============================================================
# FEATURE BUILDER
# ============================================================
def build_features_from_request(data: dict) -> pd.DataFrame:
    """
    Convert raw API request into model-ready features.
    Mirrors the feature engineering done in training.

    This function handles:
    - String parsing (int_rate, term, revol_util)
    - Missing value defaults
    - Feature engineering (ratios, flags, interactions)
    - Alignment to training feature list
    """
    d = dict(data)

    # ── Parse string fields ───────────────────────────────────
    # int_rate: handle both '13.5%' and 13.5
    ir = d.get('int_rate', 13.0)
    if isinstance(ir, str):
        ir = float(ir.replace('%', '').strip())
    d['int_rate'] = ir

    # term: handle ' 36 months', '36 months', 36
    term = d.get('term', 36)
    if isinstance(term, str):
        import re
        m = re.search(r'(\d+)', str(term))
        term = int(m.group(1)) if m else 36
    d['term'] = term

    # revol_util: handle '42.7%' and 42.7 and None
    ru = d.get('revol_util', 0.0)
    if isinstance(ru, str):
        ru = float(ru.replace('%', '').strip())
    d['revol_util'] = float(ru) if ru is not None else 0.0

    # ── Encoding maps ─────────────────────────────────────────
    GRADE_MAP = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
    SUBGRADE_MAP = {
        f"{g}{n}": (gi*5)+n
        for gi, g in enumerate(['A','B','C','D','E','F','G'])
        for n in range(1, 6)
    }
    HOME_MAP = {
        'RENT':0,'MORTGAGE':1,'OWN':2,
        'ANY':1,'OTHER':-1,'NONE':-1
    }
    VERIF_MAP = {
        'Not Verified':0,'Source Verified':1,'Verified':2
    }

    grade    = str(d.get('grade', 'C')).upper()
    subgrade = str(d.get('sub_grade', 'C3')).upper()

    d['grade_encoded']          = GRADE_MAP.get(grade, 4)
    d['sub_grade_encoded']      = SUBGRADE_MAP.get(subgrade, 18)
    d['home_ownership_encoded'] = HOME_MAP.get(
        str(d.get('home_ownership','RENT')).upper(), 0
    )
    d['verification_encoded']   = VERIF_MAP.get(
        str(d.get('verification_status','Not Verified')), 0
    )

    # ── Purpose one-hot ───────────────────────────────────────
    purpose = str(d.get('purpose', 'other')).lower().replace(' ', '_')
    PURPOSES = [
        'debt_consolidation','credit_card','home_improvement',
        'other','major_purchase','medical','small_business',
        'car','moving','vacation','house','wedding',
        'renewable_energy','educational'
    ]
    for p in PURPOSES:
        d[f'purpose_{p}'] = 1 if purpose == p else 0

    # ── Numeric defaults ──────────────────────────────────────
    annual_inc  = float(d.get('annual_inc', 60000))
    loan_amnt   = float(d.get('loan_amnt', 10000))
    installment = float(d.get('installment', 300))
    dti         = float(d.get('dti', 15.0))
    revol_bal   = float(d.get('revol_bal', 5000))
    total_acc   = float(d.get('total_acc', 15))
    open_acc    = float(d.get('open_acc', 8))
    delinq_2yrs = float(d.get('delinq_2yrs', 0))
    pub_rec     = float(d.get('pub_rec', 0))
    pub_rec_bk  = float(d.get('pub_rec_bankruptcies', 0))
    mort_acc    = float(d.get('mort_acc', 0))
    inq_6mths   = float(d.get('inq_last_6mths', 0))
    emp_years   = float(d.get('emp_length_years', 5))

    # Credit history
    try:
        issue_d  = pd.to_datetime(d.get('issue_d', 'Jan-2018'), format='%b-%Y', errors='coerce')
        earliest = pd.to_datetime(d.get('earliest_cr_line', 'Jan-2010'), format='%b-%Y', errors='coerce')
        credit_history_years = max(0.0, float((issue_d - earliest).days / 365.25))
        issue_year    = int(issue_d.year) if issue_d is not pd.NaT else 2018
        issue_quarter = int(issue_d.quarter) if issue_d is not pd.NaT else 1
    except Exception:
        credit_history_years = 8.0
        issue_year    = 2018
        issue_quarter = 1

    # ── Engineered features ───────────────────────────────────
    monthly_inc          = annual_inc / 12
    payment_to_income    = min(installment / (monthly_inc + 1), 2.0)
    loan_to_annual_income= min(loan_amnt / (annual_inc + 1), 5.0)
    total_debt_burden    = dti + payment_to_income * 100
    high_revol_util_flag = int(d['revol_util'] > 70)
    revol_util_sq        = d['revol_util'] ** 2
    derog_score          = (
        min(delinq_2yrs, 5) * 2 +
        min(pub_rec, 3)     * 3 +
        min(pub_rec_bk, 2)  * 5
    )
    credit_depth         = credit_history_years * min(total_acc, 50) / 10
    grade_dti            = d['grade_encoded'] * dti
    dti_over_30_flag     = int(dti > 30)
    log_annual_inc       = float(np.log1p(annual_inc))
    log_revol_bal        = float(np.log1p(revol_bal))
    high_risk_grade_flag = int(d['grade_encoded'] >= 5)
    open_acc_ratio       = min(open_acc / max(total_acc, 1), 1.0)

    # ── Assemble feature dict ─────────────────────────────────
    features = {
        'grade_dti_interaction'   : grade_dti,
        'int_rate'                : d['int_rate'],
        'loan_to_annual_income'   : loan_to_annual_income,
        'dti'                     : dti,
        'credit_history_years'    : credit_history_years,
        'revol_util'              : d['revol_util'],
        'credit_depth'            : credit_depth,
        'revol_bal'               : revol_bal,
        'log_revol_bal'           : log_revol_bal,
        'open_acc_ratio'          : open_acc_ratio,
        'loan_amnt'               : loan_amnt,
        'annual_inc'              : annual_inc,
        'log_annual_inc'          : log_annual_inc,
        'total_acc'               : total_acc,
        'open_acc'                : open_acc,
        'emp_length_years'        : emp_years,
        'term'                    : d['term'],
        'mort_acc'                : mort_acc,
        'issue_quarter'           : issue_quarter,
        'inq_last_6mths'          : inq_6mths,
        'derogatory_score'        : derog_score,
        'verification_encoded'    : d['verification_encoded'],
        'high_risk_grade_flag'    : high_risk_grade_flag,
        'home_ownership_encoded'  : d['home_ownership_encoded'],
        'delinq_2yrs'             : delinq_2yrs,
        'purpose_debt_consolidation': d.get('purpose_debt_consolidation', 0),
        'pub_rec'                 : pub_rec,
        'purpose_credit_card'     : d.get('purpose_credit_card', 0),
        'high_revol_util_flag'    : high_revol_util_flag,
        'pub_rec_bankruptcies'    : pub_rec_bk,
        'emp_length_missing'      : 0,
        'purpose_home_improvement': d.get('purpose_home_improvement', 0),
        'purpose_other'           : d.get('purpose_other', 0),
        'dti_over_30_flag'        : dti_over_30_flag,
    }

    # Build DataFrame aligned to training features
    row = pd.DataFrame([features])

    # Fill any missing features with 0
    for f in FEATURE_NAMES:
        if f not in row.columns:
            row[f] = 0

    # Only keep features model was trained on
    available = [f for f in FEATURE_NAMES if f in row.columns]
    return row[available]


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title       = 'Credit Risk Intelligence API',
    description = 'LendingClub-trained credit risk scoring service',
    version     = '1.0.0',
    docs_url    = '/docs',
    redoc_url   = '/redoc',
)


# ============================================================
# REQUEST SCHEMA
# ============================================================
class LoanApplication(BaseModel):
    """Input schema for a single loan application."""
    loan_amnt           : float       = Field(..., ge=500,  le=40000,  description='Loan amount requested in USD')
    term                : str         = Field('36 months',             description='Loan term: 36 months or 60 months')
    int_rate            : str         = Field('13.5%',                 description='Interest rate e.g. 13.5%')
    installment         : float       = Field(300.0,  ge=0,            description='Monthly payment amount')
    grade               : str         = Field('C',                     description='Loan grade A through G')
    sub_grade           : str         = Field('C3',                    description='Sub-grade e.g. C3')
    emp_length          : Optional[str] = Field('5 years',             description='Employment length')
    emp_length_years    : Optional[float] = Field(None, ge=0, le=11,   description='Employment years as number')
    home_ownership      : str         = Field('RENT',                  description='RENT MORTGAGE OWN OTHER')
    annual_inc          : float       = Field(..., ge=0,               description='Annual income in USD')
    verification_status : str         = Field('Not Verified',          description='Income verification level')
    purpose             : str         = Field('debt_consolidation',    description='Loan purpose')
    dti                 : float       = Field(..., ge=0, le=100,       description='Debt to income ratio')
    revol_util          : Optional[float] = Field(None, ge=0, le=100,  description='Revolving utilization percent')
    revol_bal           : float       = Field(0.0, ge=0,               description='Revolving balance USD')
    delinq_2yrs         : int         = Field(0, ge=0,                 description='Delinquencies in 2 years')
    inq_last_6mths      : int         = Field(0, ge=0,                 description='Credit inquiries last 6 months')
    open_acc            : int         = Field(8, ge=0,                 description='Open credit accounts')
    pub_rec             : int         = Field(0, ge=0,                 description='Public derogatory records')
    total_acc           : int         = Field(15, ge=0,                description='Total credit accounts')
    mort_acc            : int         = Field(0, ge=0,                 description='Mortgage accounts')
    pub_rec_bankruptcies: int         = Field(0, ge=0,                 description='Prior bankruptcies')
    issue_d             : str         = Field('Jan-2018',              description='Loan issue date e.g. Jan-2018')
    earliest_cr_line    : str         = Field('Jan-2010',              description='First credit line date')

    class Config:
        json_schema_extra = {
            'example': {
                'loan_amnt'           : 15000,
                'term'                : '36 months',
                'int_rate'            : '13.49%',
                'installment'         : 508.58,
                'grade'               : 'C',
                'sub_grade'           : 'C3',
                'emp_length'          : '5 years',
                'home_ownership'      : 'RENT',
                'annual_inc'          : 72000,
                'verification_status' : 'Source Verified',
                'purpose'             : 'debt_consolidation',
                'dti'                 : 18.4,
                'revol_util'          : 42.7,
                'revol_bal'           : 12500,
                'delinq_2yrs'         : 0,
                'inq_last_6mths'      : 1,
                'open_acc'            : 9,
                'pub_rec'             : 0,
                'total_acc'           : 22,
                'mort_acc'            : 0,
                'pub_rec_bankruptcies': 0,
                'issue_d'             : 'Jan-2018',
                'earliest_cr_line'    : 'Mar-2010',
            }
        }


class ScoreResponse(BaseModel):
    """Output schema for scoring response."""
    default_probability: float
    tier               : str
    label              : str
    action             : str
    expected_loss      : str
    color              : str
    model_version      : str
    scored_at          : str


# ============================================================
# ENDPOINTS
# ============================================================
@app.get('/health')
def health_check():
    """Quick health check for load balancers."""
    return {
        'status'       : 'healthy',
        'model_loaded' : MODEL is not None,
        'model_name'   : MODEL_NAME,
        'model_auc'    : MODEL_AUC,
        'threshold'    : OPT_THRESHOLD,
        'timestamp'    : datetime.now().isoformat(),
    }


@app.get('/model/info')
def model_info():
    """Return model metadata."""
    return {
        'model_name'    : MODEL_NAME,
        'test_roc_auc'  : MODEL_AUC,
        'threshold'     : OPT_THRESHOLD,
        'n_features'    : len(FEATURE_NAMES),
        'feature_names' : FEATURE_NAMES,
        'risk_tiers'    : {
            k: {'threshold': v['threshold'], 'action': v['action']}
            for k, v in RISK_TIERS.items()
        },
    }


@app.post('/score', response_model=ScoreResponse)
def score_loan(application: LoanApplication):
    """
    Score a single loan application.

    Returns default probability, risk tier, and recommended action.
    """
    if MODEL is None:
        raise HTTPException(
            status_code = 503,
            detail      = 'Model not loaded. Run src/training.py first.'
        )

    try:
        # Build features from request
        X = build_features_from_request(application.dict())

        # Predict
        prob = float(MODEL.predict_proba(X)[0][1])

        # Get risk tier
        tier_info = get_risk_tier(prob)

        return ScoreResponse(
            default_probability = round(prob, 4),
            tier                = tier_info['tier'],
            label               = tier_info['label'],
            action              = tier_info['action'],
            expected_loss       = tier_info['expected_loss'],
            color               = tier_info['color'],
            model_version       = '1.0.0',
            scored_at           = datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post('/score/batch')
def score_batch(applications: List[LoanApplication]):
    """
    Score multiple loan applications at once.
    Maximum batch size: 1000 applications.
    """
    if len(applications) > 1000:
        raise HTTPException(
            status_code = 400,
            detail      = 'Batch size limited to 1000 applications'
        )

    results = []
    errors  = []

    for i, app in enumerate(applications):
        try:
            result = score_loan(app)
            results.append({
                'index'              : i,
                'default_probability': result.default_probability,
                'tier'               : result.tier,
                'action'             : result.action,
            })
        except Exception as e:
            errors.append({'index': i, 'error': str(e)})

    return {
        'results'        : results,
        'errors'         : errors,
        'total_scored'   : len(results),
        'total_errors'   : len(errors),
        'scored_at'      : datetime.now().isoformat(),
    }


@app.get('/risk-tiers')
def get_risk_tiers():
    """Return all risk tier definitions."""
    return RISK_TIERS