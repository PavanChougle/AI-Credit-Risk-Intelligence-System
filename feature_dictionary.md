# Feature Dictionary — Credit Risk Intelligence

Business meaning of every feature in the model.

**Total features in final model:** 36

---

## Raw Loan Features

| Feature | Business Name | Description |
|---------|---------------|-------------|
| `loan_amnt` | Loan Size | Total amount requested by borrower in USD. |
| `int_rate` | Interest Rate | Annual interest rate on the loan (%). Higher = riskier borrower. |
| `term` | Loan Term | Loan duration in months. Either 36 or 60. |

## Raw Borrower Features

| Feature | Business Name | Description |
|---------|---------------|-------------|
| `annual_inc` | Annual Income | Self-reported annual income in USD. Right-skewed — use log transform. |
| `dti` | Debt-to-Income Ratio | Monthly debt payments / monthly gross income (%). Key risk signal. |
| `delinq_2yrs` | Delinquencies (2yr) | Number of 30+ day delinquencies in past 2 years. |
| `inq_last_6mths` | Credit Inquiries (6mo) | Number of hard credit pulls in last 6 months. High count = credit-seeking behavior. |
| `open_acc` | Open Accounts | Number of currently open credit lines. |
| `pub_rec` | Public Records | Number of derogatory public records (liens, judgments). |
| `revol_bal` | Revolving Balance | Total outstanding revolving credit balance in USD. |
| `revol_util` | Revolving Utilization | Revolving credit used / revolving credit limit (%). >70% is a red flag. |
| `total_acc` | Total Accounts | Total number of credit lines ever opened. |
| `mort_acc` | Mortgage Accounts | Number of mortgage accounts. Homeowners tend to be more stable. |
| `pub_rec_bankruptcies` | Bankruptcies | Number of public record bankruptcies. Severe derogatory event. |
| `emp_length_years` | Employment Length | Years at current employer (0-10+). Parsed from text. Longer = more stable income. |

## Time Features

| Feature | Business Name | Description |
|---------|---------------|-------------|
| `credit_history_years` | Credit History Length | Years between earliest credit line and loan issue date. Longer = more experienced borrower = lower risk. |
| `issue_year` | Issue Year | Year loan was issued. Used for vintage analysis and temporal train/test split. |
| `issue_quarter` | Issue Quarter | Quarter loan was issued (1-4). Captures seasonal lending patterns. |

## Encoded Categorical

| Feature | Business Name | Description |
|---------|---------------|-------------|
| `home_ownership_encoded` | Home Ownership (Encoded) | Housing stability: OWN=2, MORTGAGE=1, RENT=0, OTHER=-1. |
| `verification_encoded` | Verification Level (Encoded) | Income verification: Verified=2, Source Verified=1, Not Verified=0. |

## Engineered Domain Features

| Feature | Business Name | Description |
|---------|---------------|-------------|
| `loan_to_annual_income` | Loan-to-Income Ratio | Loan amount / annual income. Measures loan size relative to earnings capacity. |
| `derogatory_score` | Derogatory Score | Weighted composite of negative credit events. delinq*2 + pub_rec*3 + bankruptcy*5. Higher = worse credit behavior. |
| `credit_depth` | Credit Depth | credit_history_years x total_acc / 10. Experienced + broad credit profile = lower risk. |
| `grade_dti_interaction` | Grade x DTI Interaction | grade_encoded * dti. Captures mispriced risk: a Grade B borrower with high DTI is riskier than priced. |
| `log_annual_inc` | Log Annual Income | log1p(annual_inc). Log transform corrects right-skew. More useful for linear models. |
| `high_risk_grade_flag` | High Risk Grade Flag | Binary: 1 if grade is E, F, or G. These grades have meaningfully higher default profiles. |
| `open_acc_ratio` | Open Account Ratio | open_acc / total_acc. Low ratio may indicate many closed or defaulted accounts. |

## Missing Value Indicators

| Feature | Business Name | Description |
|---------|---------------|-------------|
| `emp_length_missing` | Employment Missing Flag | Binary: 1 if emp_length was not reported. Self-employed or gig workers often omit this. |

## Loan Purpose (One-Hot)

| Feature | Business Name | Description |
|---------|---------------|-------------|
| `purpose_small_business` | Purpose: Small Business | HIGHEST default rate (30-40%). High variance — hardest segment to model. |
| `purpose_credit_card` | Purpose: Credit Card | Below-average default rate. Financially disciplined borrowers paying off cards. |
| `purpose_debt_consolidation` | Purpose: Debt Consolidation | Largest segment. Near-average default rate. Borrowers combining existing debts. |
| `purpose_other` | Purpose: Other | Catch-all category. Average default rate. |
| `purpose_car` | Purpose: Car | Below-average default. Secured by asset. |
| `purpose_home_improvement` | Purpose: Home Improvement | Below-average default rate. Homeowners investing in property. |
| `purpose_major_purchase` | Purpose: Major Purchase | Near-average default rate. |

## Target Variable

| Feature | Business Name | Description |
|---------|---------------|-------------|
| `target` | Target (Default) | Binary outcome: 1=Default/Charged Off, 0=Fully Paid. This is what we predict. |

---
*Generated automatically by src/feature_engineering.py*
