# Credit Risk Classification Model

## Overview
This project implements a comprehensive machine learning pipeline for credit risk classification. The model analyzes credit applicant data through rigorous feature selection, engineering, and evaluation to predict credit approval categories (P1, P2, P3, P4) with **78% accuracy**.

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset Information](#dataset-information)
- [Pipeline Architecture](#pipeline-architecture)
- [Model Performance](#model-performance)
- [Usage](#usage)
- [Output Interpretation](#output-interpretation)
- [Technical Details](#technical-details)
- [Business Insights](#business-insights)
- [Troubleshooting](#troubleshooting)

---

## Project Structure

```
credit-risk-classification/
│
├── dataset/
│   ├── case_study1.xlsx          # Trade line information (26 columns)
│   └── case_study2.xlsx          # Credit behavior & demographics (62 columns)
│
├── train_credit_risk_model.py   # Main training script
├── README.md                      # This file
└── Results.txt                    # Sample output (optional)
```

---

## Requirements

### System Requirements
- **Python**: 3.7 or higher
- **RAM**: Minimum 8GB recommended
- **Storage**: ~500MB for datasets and dependencies

### Python Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn scipy statsmodels xgboost openpyxl
```

**Specific versions tested:**
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- statsmodels >= 0.13.0
- scipy >= 1.7.0
- openpyxl >= 3.0.9

**Quick install:**
```bash
pip install -r requirements.txt
```

---

## Dataset Information

### Input Files

#### 1. case_study1.xlsx (Trade Line Data)
**Records**: 51,336  
**Columns**: 26  

**Key Features:**
- **Account Types**: Auto_TL, CC_TL, Consumer_TL, Gold_TL, Home_TL, PL_TL, Secured_TL, Unsecured_TL, Other_TL
- **Trade Line Metrics**: Total_TL, Tot_Active_TL, Tot_Closed_TL
- **Time Windows**: L6M (Last 6 Months), L12M (Last 12 Months)
- **Account Age**: Age_Oldest_TL, Age_Newest_TL
- **Payment History**: Tot_Missed_Pmnt

#### 2. case_study2.xlsx (Credit Behavior & Demographics)
**Records**: 51,336  
**Columns**: 62  

**Key Features:**
- **Delinquency Metrics**: 
  - num_times_delinquent, max_delinquency_level
  - num_deliq_6mts, num_deliq_12mts
  - num_times_30p_dpd, num_times_60p_dpd
  
- **Account Status**:
  - Standard (num_std), Substandard (num_sub)
  - Doubtful (num_dbt), Loss (num_lss)
  
- **Enquiry Patterns**:
  - Credit Card enquiries (CC_enq, CC_enq_L6m, CC_enq_L12m)
  - Personal Loan enquiries (PL_enq, PL_enq_L6m, PL_enq_L12m)
  - Total enquiries (tot_enq, enq_L3m, enq_L6m, enq_L12m)
  
- **Demographics**:
  - AGE, GENDER, MARITALSTATUS, EDUCATION
  - NETMONTHLYINCOME, Time_With_Curr_Empr
  
- **Utilization Metrics**:
  - CC_utilization, PL_utilization
  - pct_currentBal_all_TL
  - max_unsec_exposure_inPct
  
- **Product Information**:
  - last_prod_enq2, first_prod_enq2
  - CC_Flag, PL_Flag, HL_Flag, GL_Flag
  
- **Target Variable**: **Approved_Flag** (P1, P2, P3, P4)
- **Credit Score**: Numerical credit score

### Data Cleaning Summary
- **Missing Value Indicator**: -99999
- **Initial Records**: 51,336
- **Records After Cleaning**: 42,064
- **Null Values After Merge**: 0
- **Columns Removed**: Columns with >10,000 missing values

---

## Pipeline Architecture

### 1. Data Loading & Preprocessing
```
Raw Data (51,336 records) 
    ↓
Remove -99999 values from df1['Age_Oldest_TL']
    ↓
Drop columns with >10,000 missing values from df2
    ↓
Remove all rows containing -99999 in df2
    ↓
Inner join on PROSPECTID
    ↓
Clean Dataset (42,064 records, 0 nulls)
```

### 2. Feature Selection Process

#### Step 2.1: Chi-Square Test (Categorical Features)
Tests independence between categorical variables and target:

| Feature | P-Value | Result |
|---------|---------|--------|
| MARITALSTATUS | 3.58e-233 | ✅ Highly Significant |
| EDUCATION | 2.69e-30 | ✅ Highly Significant |
| GENDER | 1.91e-05 | ✅ Significant |
| last_prod_enq2 | 0.0 | ✅ Extremely Significant |
| first_prod_enq2 | 7.85e-287 | ✅ Extremely Significant |

**All categorical features retained** (p-value < 0.05)

#### Step 2.2: VIF Analysis (Multicollinearity Check)
**Threshold**: VIF ≤ 6

**Process**:
- Sequential elimination of features with VIF > 6
- Iterative recalculation after each removal
- **Result**: 37 numerical features retained (from ~80 original)

**Features with High VIF (Removed)**:
- Features with inf VIF (perfect multicollinearity)
- Features with VIF > 100 (extreme multicollinearity)
- Redundant time-window features

**Example VIF Values** (Final Selected Features):
```
pct_tl_open_L6M:              5.15
pct_tl_closed_L6M:            2.61
Tot_TL_closed_L12M:           3.83
pct_tl_closed_L12M:           5.58
Tot_Missed_Pmnt:              1.99
CC_TL:                        4.81
Home_TL:                      4.38
PL_TL:                        3.06
```

#### Step 2.3: ANOVA Test (Numerical Features)
**Purpose**: Test if feature means differ across approval classes  
**Threshold**: p-value ≤ 0.05  

**Last Feature Tested**:
- F-statistic: 507.29
- P-value: 5e-324

**All VIF-selected features passed ANOVA** → Final 37 numerical features

### 3. Feature Engineering

#### 3.1 Ordinal Encoding (EDUCATION)
```python
SSC            → 1
12TH           → 2
GRADUATE       → 3
UNDER GRADUATE → 3
POST-GRADUATE  → 4
OTHERS         → 1
PROFESSIONAL   → 3
```

**Distribution After Encoding**:
```
3 (Graduate-level):     ~60%
2 (High School):        ~25%
4 (Post-Graduate):      ~10%
1 (Below High School):  ~5%
```

#### 3.2 One-Hot Encoding
**Applied to:**
- MARITALSTATUS → `MARITALSTATUS_Married`, `MARITALSTATUS_Single`
- GENDER → `GENDER_M`, `GENDER_F`
- last_prod_enq2 → 6 binary columns (PL, AL, CC, HL, ConsumerLoan, others)
- first_prod_enq2 → 6 binary columns (PL, AL, CC, HL, ConsumerLoan, others)

**Final Feature Count**: 55 columns
- 37 numerical features
- 1 ordinal feature (EDUCATION)
- 16 one-hot encoded features
- 1 target variable (Approved_Flag)

#### 3.3 Feature Scaling (StandardScaler)
**Scaled Features** (for XGBoost with scaling):
```
Age_Oldest_TL
Age_Newest_TL
time_since_recent_payment
max_recent_level_of_deliq
recent_level_of_deliq
time_since_recent_enq
NETMONTHLYINCOME
Time_With_Curr_Empr
```

**Formula**: `z = (x - μ) / σ`

### 4. Model Training

Three classifiers evaluated:

#### 4.1 Random Forest
```python
n_estimators = 200
random_state = 42
test_size = 0.2
```

#### 4.2 XGBoost
```python
objective = 'multi:softmax'
num_class = 4
test_size = 0.2
random_state = 42
```

#### 4.3 Decision Tree
```python
max_depth = 20
min_samples_split = 10
test_size = 0.2
random_state = 42
```

### 5. Hyperparameter Tuning (XGBoost)

**Method**: GridSearchCV with 3-fold cross-validation

**Parameter Grid**:
```python
{
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
```

**Best Parameters Found**:
```python
{
    'learning_rate': 0.1,
    'max_depth': 5,
    'n_estimators': 200
}
```

---

## Model Performance

### Overall Accuracy Comparison

| Model | Accuracy | Ranking |
|-------|----------|---------|
| **XGBoost** | **78.0%** | 🥇 Best |
| **XGBoost + Scaling** | **78.0%** | 🥇 Best |
| **Random Forest** | 76.4% | 🥈 2nd |
| **Decision Tree** | 71.0% | 🥉 3rd |
| **XGBoost (Tuned)** | **77.8%** | - |

### Detailed Performance Metrics

#### Random Forest (76.4% Accuracy)

| Class | Precision | Recall | F1-Score | Interpretation |
|-------|-----------|--------|----------|----------------|
| **P1** | 83.7% | 70.4% | 76.5% | High precision, moderate recall |
| **P2** | 79.6% | **92.8%** | **85.7%** | Best overall performance |
| **P3** | 44.2% | 21.1% | 28.6% | Challenging class |
| **P4** | 71.8% | 72.7% | 72.2% | Balanced performance |

#### XGBoost (78.0% Accuracy) ⭐ BEST MODEL

| Class | Precision | Recall | F1-Score | Interpretation |
|-------|-----------|--------|----------|----------------|
| **P1** | 82.6% | 76.3% | 79.3% | Strong balanced performance |
| **P2** | 82.6% | **91.5%** | **86.8%** | Excellent recall |
| **P3** | 46.6% | 30.7% | 37.0% | Improved over RF |
| **P4** | 73.1% | 71.9% | 72.5% | Consistent performance |

#### Decision Tree (71.0% Accuracy)

| Class | Precision | Recall | F1-Score | Interpretation |
|-------|-----------|--------|----------|----------------|
| **P1** | 72.4% | 72.4% | 72.4% | Perfectly balanced |
| **P2** | 80.9% | 82.5% | 81.6% | Good performance |
| **P3** | 34.6% | 33.0% | 33.8% | Weakest class |
| **P4** | 65.2% | 62.8% | 64.0% | Moderate performance |

#### XGBoost with Feature Scaling (78.0% Accuracy)

| Class | Precision | Recall | F1-Score | Notes |
|-------|-----------|--------|----------|-------|
| **P1** | 82.6% | 76.3% | 79.3% | Identical to XGBoost |
| **P2** | 82.6% | 91.5% | 86.8% | No improvement from scaling |
| **P3** | 46.6% | 30.7% | 37.0% | Same as XGBoost |
| **P4** | 73.1% | 71.9% | 72.5% | Same as XGBoost |

**Finding**: Feature scaling does not improve XGBoost (tree-based models are scale-invariant)

#### XGBoost Hyperparameter Tuned (77.8% Accuracy)

**Best Hyperparameters**:
```
learning_rate: 0.1
max_depth: 5
n_estimators: 200
```

**Test Accuracy**: 77.8%

**Note**: Slight decrease from default (78.0%) suggests default parameters are well-suited or overfitting on CV folds.

---

## Usage

### Basic Execution

```bash
python train_credit_risk_model.py
```

### Expected Runtime

| Stage | Duration | Notes |
|-------|----------|-------|
| Data Loading | 5-10 sec | Depends on Excel file size |
| Data Cleaning | 10-15 sec | Row-wise operations |
| VIF Calculation | 1-2 min | Most time-consuming |
| ANOVA Tests | 10-15 sec | 37 features × 4 groups |
| Model Training (RF) | 30-45 sec | 200 trees |
| Model Training (XGBoost) | 15-20 sec | Gradient boosting |
| Model Training (DT) | 5-10 sec | Single tree |
| Hyperparameter Tuning | 5-10 min | 27 combinations × 3 folds |
| **Total** | **~10-15 min** | Full pipeline |

### Command-Line Options

```bash
# Run with output redirection
python train_credit_risk_model.py > output.txt 2>&1

# Run in background (Linux/Mac)
nohup python train_credit_risk_model.py > output.log 2>&1 &

# Run with Python profiler
python -m cProfile -o profile.stats train_credit_risk_model.py
```

---

## Output Interpretation

### Section-by-Section Guide

#### 1. Data Information
```
================================================================================
DF1 INFO
================================================================================
<class 'pandas.DataFrame'>
RangeIndex: 51336 entries, 0 to 51335
Data columns (total 26 columns):
...
```
**What to check**: Column counts, data types, memory usage

#### 2. Common Columns
```
================================================================================
COMMON COLUMN NAMES BETWEEN DF1 AND DF2
================================================================================
PROSPECTID
```
**What to check**: Only PROSPECTID should appear (merge key)

#### 3. Chi-Square Results
```
MARITALSTATUS --- 3.578180861038862e-233
EDUCATION --- 2.6942265249737532e-30
...
```
**Interpretation**: All p-values << 0.05 → all categorical features are significant

#### 4. VIF Sequential Check
```
0 --- inf        ← Remove (perfect multicollinearity)
0 --- inf        ← Remove again
0 --- 11.32      ← Remove (VIF > 6)
0 --- 8.36       ← Remove (VIF > 6)
0 --- 6.52       ← Remove (VIF > 6)
0 --- 5.15       ← Keep (VIF ≤ 6) ✅
1 --- 2.61       ← Keep ✅
```
**What to watch**: Features with VIF ≤ 6 are retained

#### 5. Model Performance Output
```
================================================================================
XGBOOST CLASSIFIER
================================================================================

Accuracy: 0.78

Class p1:
Precision: 0.8260405549626467
Recall: 0.7633136094674556
F1 Score: 0.7934392619169657
...
```

**Metrics Explained**:
- **Accuracy**: Overall correct predictions (78%)
- **Precision**: Of predicted P1, how many are actually P1? (82.6%)
- **Recall**: Of actual P1, how many did we predict? (76.3%)
- **F1 Score**: Harmonic mean of precision and recall (79.3%)

---

## Technical Details

### Final Feature Set (55 Total)

#### Numerical Features (37)
```
pct_tl_open_L6M, pct_tl_closed_L6M
Tot_TL_closed_L12M, pct_tl_closed_L12M
Tot_Missed_Pmnt
CC_TL, Home_TL, PL_TL, Secured_TL, Unsecured_TL, Other_TL
Age_Oldest_TL, Age_Newest_TL
time_since_recent_payment
max_recent_level_of_deliq
num_deliq_6_12mts
num_times_60p_dpd
num_std_12mts
num_sub, num_sub_6mts, num_sub_12mts
num_dbt, num_dbt_12mts
num_lss
recent_level_of_deliq
CC_enq_L12m, PL_enq_L12m
time_since_recent_enq
enq_L3m
NETMONTHLYINCOME
Time_With_Curr_Empr
CC_Flag, PL_Flag
pct_PL_enq_L6m_of_ever, pct_CC_enq_L6m_of_ever
HL_Flag, GL_Flag
```

#### Categorical Features (18 → 17 after encoding)
```
EDUCATION (ordinal: 1-4)
MARITALSTATUS_Married, MARITALSTATUS_Single
GENDER_M, GENDER_F
last_prod_enq2_PL, last_prod_enq2_AL, last_prod_enq2_CC, 
last_prod_enq2_HL, last_prod_enq2_ConsumerLoan, last_prod_enq2_others
first_prod_enq2_PL, first_prod_enq2_AL, first_prod_enq2_CC,
first_prod_enq2_HL, first_prod_enq2_ConsumerLoan, first_prod_enq2_others
```

### Train-Test Split
- **Training Set**: 80% (33,651 records)
- **Test Set**: 20% (8,413 records)
- **Random State**: 42 (reproducible results)

### Model Configurations

#### Random Forest Configuration
```python
RandomForestClassifier(
    n_estimators=200,      # 200 decision trees
    random_state=42,       # Reproducibility
    # Other params use sklearn defaults:
    # max_depth=None (unlimited)
    # min_samples_split=2
    # min_samples_leaf=1
)
```

#### XGBoost Configuration
```python
XGBClassifier(
    objective='multi:softmax',  # Multi-class classification
    num_class=4,                # 4 classes (P1, P2, P3, P4)
    # Other params use xgboost defaults:
    # learning_rate=0.3
    # max_depth=6
    # n_estimators=100
)
```

#### Decision Tree Configuration
```python
DecisionTreeClassifier(
    max_depth=20,           # Prevent overfitting
    min_samples_split=10,   # Require 10 samples to split
    # Other params use sklearn defaults
)
```

---

## Business Insights

### Class Interpretation

| Class | Risk Level | Approval Strategy | Model Confidence |
|-------|------------|-------------------|------------------|
| **P1** | Low Risk | ✅ Approve with standard terms | High (82.6% precision) |
| **P2** | Moderate-Low Risk | ✅ Approve with monitoring | Very High (91.5% recall) |
| **P3** | High Risk | ⚠️ Manual review required | Low (46.6% precision) |
| **P4** | Moderate-High Risk | 🔍 Enhanced due diligence | Moderate (73.1% precision) |

### Key Predictive Features

Based on VIF retention and ANOVA significance:

**Top Delinquency Indicators**:
- max_recent_level_of_deliq
- num_deliq_6_12mts
- num_times_60p_dpd
- recent_level_of_deliq

**Top Account Behavior Indicators**:
- Tot_Missed_Pmnt
- pct_tl_closed_L6M / L12M
- CC_TL, PL_TL (credit card and personal loan accounts)

**Top Enquiry Indicators**:
- CC_enq_L12m, PL_enq_L12m
- enq_L3m (recent enquiries indicate credit-seeking behavior)
- last_prod_enq2, first_prod_enq2 (product enquiry patterns)

**Top Financial Indicators**:
- NETMONTHLYINCOME
- Time_With_Curr_Empr (employment stability)

### Business Recommendations

#### 1. For Class P3 (Challenging Class)
**Problem**: Low precision (46.6%) and recall (30.7%)

**Possible Causes**:
- Class imbalance (fewer P3 samples)
- Overlapping characteristics with P2/P4
- Insufficient discriminative features

**Recommendations**:
- Collect more P3 samples
- Engineer P3-specific features
- Use SMOTE or class weighting
- Consider binary classification (P3 vs. non-P3)
- Implement ensemble voting with P3 specialist model

#### 2. Automated Decision Rules
```
IF precision >= 80% AND recall >= 75%:
    → Automate approval/rejection
    
ELSE IF precision >= 70% OR recall >= 70%:
    → Flag for quick manual review
    
ELSE:
    → Detailed manual underwriting
```

**Application**:
- **P1, P2**: Automate (high confidence)
- **P4**: Quick review (moderate confidence)
- **P3**: Detailed review (low confidence)

#### 3. Risk-Adjusted Pricing
Use model probabilities for interest rate determination:

```python
base_rate = 10%
risk_premium = (1 - model_confidence) * 5%
final_rate = base_rate + risk_premium
```

#### 4. Monitoring & Model Drift
- Track model accuracy monthly
- Retrain when accuracy drops below 75%
- Monitor feature importance changes
- Track class distribution shifts

---

## Advanced Usage

### 1. Save Trained Model
Add to script after training:
```python
import pickle

# Save XGBoost model
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_classifier, f)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
```

### 2. Load and Predict
```python
import pickle
import pandas as pd

# Load model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load encoder
with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Make predictions
new_data = pd.read_excel('new_applicants.xlsx')
# ... (apply same preprocessing) ...
predictions = model.predict(new_data)
predicted_classes = encoder.inverse_transform(predictions)
```

### 3. Feature Importance Analysis
```python
import matplotlib.pyplot as plt

# Get feature importances
importances = xgb_classifier.feature_importances_
feature_names = x_train.columns

# Create DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Plot top 20
importance_df.head(20).plot(x='feature', y='importance', kind='barh', figsize=(10, 8))
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

### 4. Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate predictions
y_pred = xgb_classifier.predict(x_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['P1', 'P2', 'P3', 'P4'],
            yticklabels=['P1', 'P2', 'P3', 'P4'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - XGBoost')
plt.savefig('confusion_matrix.png')
```

---

## Future Enhancements

### 1. Handle Class Imbalance
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
```

### 2. Ensemble Methods
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_classifier),
        ('xgb', xgb_classifier),
        ('dt', dt_model)
    ],
    voting='soft'  # Use probability averaging
)
ensemble.fit(x_train, y_train)
```

### 3. Cross-Validation
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(xgb_classifier, x, y_encoded, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
```

### 4. Hyperparameter Tuning with RandomizedSearchCV
```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [50, 100, 150, 200, 250],
    'max_depth': range(3, 10),
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(
    xgb_model, 
    param_distributions, 
    n_iter=50,  # Try 50 random combinations
    cv=3, 
    scoring='accuracy', 
    n_jobs=-1,
    random_state=42
)
random_search.fit(x_train, y_train)
```

---

## License
This project is for educational and analytical purposes.

---

**Last Updated**: February 2026  
**Version**: 1.0  
**Status**: Production-Ready ✅
