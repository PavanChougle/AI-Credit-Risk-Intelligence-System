# 💳 Ai Credit Risk Intelligence System

### LendingClub Portfolio Analysis | 1.3M+ Loans | 2012–2018

## app:https://ai-credit-risk-intelligence-system-zg8wvqgitqgwubsnyo5pyw.streamlit.app/

!\[Python](https://img.shields.io/badge/Python-3.10-blue)

!\[Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

!\[LightGBM](https://img.shields.io/badge/Model-LightGBM-green)

!\[Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)



---



## 🎯 Problem Statement



Consumer lending institutions lose billions annually from loan defaults.

Traditional rule-based credit scoring misclassifies high-risk borrowers,

leading to preventable losses. This system builds an ML-powered credit

risk engine that quantifies default probability, segments portfolios into

4 risk tiers, and quantifies financial impact using Basel III methodology.



\*\*Business Question:\*\*

> "How much money can we save by replacing gut-feel approvals

>  with a data-driven risk model?"



---



## 📊 Portfolio at a Glance



| Metric           | Value         |

|------------------|---------------|

| Total Loans      | \*\*1,303,638\*\* |

| Total Defaults   | \*\*261,686\*\*   |

| Good Loans       | \*\*1,041,952\*\* |

| Default Rate     | \*\*20.1%\*\*     |

| Average Loan Size| \*\*$14,417\*\*   |



---



## 💡 Key Discoveries



- 🔴 \*\*Grade G loans default at 50.1%\*\* — 8x higher than

&#x20; Grade A (6.1%), yet share similar approval volumes

- 📈 \*\*Interest rate is the #1 predictive feature\*\* (score: 476),

&#x20; followed by annual income (454) and issue year (443)

- 📉 \*\*DTI cliff at 30%\*\* — default rates jump sharply above

&#x20; 30% DTI, validating the threshold as a hard cutoff rule

- 💊 \*\*Loan-to-income ratio\*\* is a stronger signal than raw

&#x20; loan amount or income alone (engineered feature)

- ⚡ \*\*Grade × DTI interaction\*\* ranks in top 10 features,

&#x20; confirming combined risk factors are non-linear



---



## 🏗️ Technical Architecture



Raw Data (1.3M loans)

│

▼

┌─────────────────────┐

│ Data Pipeline │ ← Missing value imputation

│ (Preprocessing) │ Outlier capping ($250k income)

└────────┬────────────┘ Feature engineering (50+ features)

│

▼

┌─────────────────────┐

│ Feature Engineering│ ← loan\_to\_annual\_income ratio

│ │ grade\_dti\_interaction

└────────┬────────────┘ credit\_history\_years

│ derogatory\_score

▼

┌──────────────────────────────┐

│ Model Training │

│ ├── Logistic Regression │ ← Baseline

│ └── LightGBM │ ← Primary model

└────────┬─────────────────────┘

│

▼

┌─────────────────────┐

│ Risk Tier Engine │ ← ACCEPT / REVIEW / CAUTION / DECLINE

│ (Basel III EL) │ EL = PD × LGD × EAD

└────────┬────────────┘ LGD = 70% (unsecured loans)

│

▼

┌─────────────────────┐

│ Streamlit Dashboard │ ← Portfolio Analytics

│ │ Model Performance

└─────────────────────┘ Risk Tier Configuration

Live Loan Scorer





---



## 📊 Model Performance



| Metric | Logistic Regression | LightGBM |

|--------|:-------------------:|:--------:|

| ROC-AUC | 0.6953 | 0.6936 |

| PR-AUC | 0.3803 | 0.3887 |

| Optimal Threshold | 0.505 | 0.247 |



### Confusion Matrix — Logistic Regression (Threshold = 0.50)



| | Predicted Good | Predicted Default |

|---|:--------------:|:-----------------:|

| \*\*Actual Good\*\* | 98,136 ✅ TN | 64,625 ❌ FP (LOSS) |

| \*\*Actual Default\*\* | 13,849 ⚠️ FN (MISSED REVENUE) | 29,491 ✅ TP |



---



## 🎯 Risk Tier Framework (4-Tier System)



| Tier | PD Range | Action | Portfolio % | Expected Loss |

|------|:--------:|--------|:-----------:|:-------------:|

| 🟢 \*\*ACCEPT\*\* | 0% – 12% | Auto-approve, no review | \~35% | < 3% |

| 🟡 \*\*REVIEW\*\* | 12% – 25% | Route to underwriter | \~30% | 3% – 8% |

| 🟠 \*\*CAUTION\*\* | 25% – 40% | Reduced amount / co-signer | \~20% | 8% – 15% |

| 🔴 \*\*DECLINE\*\* | 40%+ | ECOA-compliant decline | \~15% | > 15% |



> \*\*ECOA Compliance:\*\* All DECLINE decisions include top 3

> SHAP-powered decline reasons as required by the Equal

> Credit Opportunity Act (ECOA) and Fair Credit Reporting Act (FCRA)



---



## 🔍 Top Features by Importance



| Rank | Feature | Score | Category |

|------|---------|------:|----------|

| 1 | int\_rate | 476 | Loan Terms |

| 2 | annual\_inc | 454 | Income/Loan Size |

| 3 | issue\_year | 443 | Other |

| 4 | revol\_bal | 426 | Revolving Utilization |

| 5 | loan\_to\_annual\_income | 388 | Income/Loan Size |

| 6 | credit\_history\_years | 348 | Credit History |

| 7 | revol\_util | 312 | Revolving Utilization |

| 8 | loan\_amnt | 307 | Income/Loan Size |

| 9 | grade\_dti\_interaction | 300 | Grade Features |

| 10 | open\_acc | 272 | Credit History |



\---



## 💰 Financial Impact (Basel III Framework)



EL = PD × LGD × EAD

LGD = 70% (unsecured personal loans — industry standard)



Scenario A — No Model (approve everyone):

All 261,686 defaults × $14,417 avg × 70% LGD



Scenario B — With Model (decline DECLINE tier):

Filtered portfolio — bottom 15% declined



Result: Measurable reduction in expected credit losses





---



## 📁 Project Structure



credit-risk-intelligence/

├── data/

│ ├── raw/                     # Original CSV

│ └── processed/               # Parquet files

├── src/

│   ├── preprocessing.py       # Data cleaning pipeline

│   ├── features.py            # Feature engineering

│   ├── training.py            # Model training + evaluation

│   └── monitoring.py          # Drift detection

├── app/

│   ├── api.py                 # FastAPI endpoint

│   └── dashboard.py           # Streamlit dashboard

├── models

│   └── credit\_risk\_v1.pkl

├── notebook/

│   └──eda_insight.ipynb

├── outputs/

│   ├── risk\_segments.png # borrower risk segmentation

│   ├── correlation\_matrix.png  feature relationship map

│   ├── feature\_importance\_report.png

│   ├── shap\_explanation.png

│   ├── model\_comparison.csv # Model metrics

│   ├── model\_evaluation.png # ROC, PR, Calibration

│   ├── caliberation\_plot.png

│   ├── confusion\_matrix.png

│   ├── phase1\_validation.png

│   ├── grade\_risk\_analysis.png

│   ├── drift\_dashboard.png

│   ├── roc\_pr\_curves.png

│   ├── threshold\_analysis.png

│   └── purpose\_default\_rates.png      

├── feature_dictionary.md

├── deployment_guide.md/

├── README.md
└── requirements.txt  





---



## 🚀 How to Run



bash

# 1. Clone repo

git clone https://github.com/yourname/credit-risk-intelligence

cd credit-risk-intelligence



# 2. Install dependencies

pip install -r requirements.txt



# 3. Add data

# Place lending\_club\_loans.csv in data/raw/



# 4. Run the pipeline (in order)
python src/preprocessing.py
python src/features.py
python src/training.py

# 5. Start the API (optional — separate terminal)
uvicorn app.api:app --reload

# 6. Launch dashboard
streamlit run app/dashboard.py

---

👤 Author
Pavan Chougle

💼 https://www.linkedin.com/in/pavan-chougle
🐙 https://github.com/PavanChougle

Built for portfolio demonstration.
Data: LendingClub public dataset via Kaggle.
https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv
