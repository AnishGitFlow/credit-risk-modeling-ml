# ============================================================================
# Credit Risk Classification Model
# ============================================================================

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, accuracy_score, classification_report, precision_recall_fscore_support
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# Data Loading
# ============================================================================

a1 = pd.read_excel("./dataset/case_study1.xlsx")
a2 = pd.read_excel("./dataset/case_study2.xlsx")

df1 = a1.copy()
df2 = a2.copy()

print("=" * 80)
print("DF1 INFO")
print("=" * 80)
df1.info()

print("\n" + "=" * 80)
print("DF2 INFO")
print("=" * 80)
df2.info()

# ============================================================================
# Data Cleaning
# ============================================================================

# Remove nulls from df1
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

# Identify columns to remove from df2
columns_to_be_removed = []
for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)

# Remove identified columns
df2 = df2.drop(columns_to_be_removed, axis=1)

# Remove rows with -99999 values
for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]

# Checking common column names
print("\n" + "=" * 80)
print("COMMON COLUMN NAMES BETWEEN DF1 AND DF2")
print("=" * 80)
for i in list(df1.columns):
    if i in list(df2.columns):
        print(i)

# ============================================================================
# Data Merging
# ============================================================================

df = pd.merge(df1, df2, how='inner', left_on=['PROSPECTID'], right_on=['PROSPECTID'])

print("\n" + "=" * 80)
print("NULL VALUES CHECK AFTER MERGE")
print("=" * 80)
print(df.isna().sum().sum())

# Check categorical columns
print("\n" + "=" * 80)
print("CATEGORICAL COLUMNS")
print("=" * 80)
for i in df.columns:
    if df[i].dtype == 'object':
        print(i)

# ============================================================================
# Feature Selection - Chi-Square Test
# ============================================================================

print("\n" + "=" * 80)
print("CHI-SQUARE TEST P-VALUES")
print("=" * 80)
for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i, '---', pval)

# ============================================================================
# Feature Selection - VIF Analysis
# ============================================================================

# Identify numerical columns
# Identify numerical columns STRICTLY (statsmodels-safe)
numeric_columns = [
    col for col in df.columns
    if col not in ['PROSPECTID', 'Approved_Flag']
    and pd.api.types.is_numeric_dtype(df[col])
]

# VIF sequential check
print("\n" + "=" * 80)
print("VIF SEQUENTIAL CHECK")
print("=" * 80)
vif_data = df[numeric_columns].astype(float).copy()
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range(0, total_columns):
    vif_value = variance_inflation_factor(vif_data, column_index)
    print(column_index, '---', vif_value)
    
    if vif_value <= 6:
        columns_to_be_kept.append(numeric_columns[i])
        column_index = column_index + 1
    else:
        vif_data = vif_data.drop([numeric_columns[i]], axis=1)

# ============================================================================
# Feature Selection - ANOVA Test
# ============================================================================

columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])
    b = list(df['Approved_Flag'])
    
    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']
    
    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)
    
    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)

print("\n" + "=" * 80)
print("LAST ANOVA F-STATISTIC AND P-VALUE")
print("=" * 80)
print("F-statistic:", f_statistic)
print("P-value:", p_value)

# Final feature list
features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]

# ============================================================================
# Feature Encoding
# ============================================================================

print("\n" + "=" * 80)
print("UNIQUE VALUES IN CATEGORICAL COLUMNS")
print("=" * 80)
print("MARITALSTATUS unique values:", df['MARITALSTATUS'].unique())
print("EDUCATION unique values:", df['EDUCATION'].unique())
print("GENDER unique values:", df['GENDER'].unique())
print("last_prod_enq2 unique values:", df['last_prod_enq2'].unique())
print("first_prod_enq2 unique values:", df['first_prod_enq2'].unique())

# Ordinal encoding for EDUCATION
education_map = {
    "SSC": 1,
    "12TH": 2,
    "GRADUATE": 3,
    "UNDER GRADUATE": 3,
    "POST-GRADUATE": 4,
    "OTHERS": 1,
    "PROFESSIONAL": 3
}

df["EDUCATION"] = df["EDUCATION"].map(education_map).astype("int64")

print("\n" + "=" * 80)
print("EDUCATION VALUE COUNTS AFTER ENCODING")
print("=" * 80)
print(df['EDUCATION'].value_counts())

df['EDUCATION'] = df['EDUCATION'].astype(int)

print("\n" + "=" * 80)
print("DF INFO AFTER EDUCATION ENCODING")
print("=" * 80)
df.info()

# One-hot encoding for categorical features
df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])

print("\n" + "=" * 80)
print("DF_ENCODED INFO")
print("=" * 80)
df_encoded.info()

print("\n" + "=" * 80)
print("DF_ENCODED DESCRIBE")
print("=" * 80)
print(df_encoded.describe())

# ============================================================================
# Model Training - Random Forest
# ============================================================================

print("\n" + "=" * 80)
print("RANDOM FOREST CLASSIFIER")
print("=" * 80)

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
rf_classifier.fit(x_train, y_train)
y_pred = rf_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Accuracy: {accuracy}')
print()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

# ============================================================================
# Model Training - XGBoost
# ============================================================================

print("=" * 80)
print("XGBOOST CLASSIFIER")
print("=" * 80)

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis=1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Accuracy: {accuracy:.2f}')
print()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

# ============================================================================
# Model Training - Decision Tree
# ============================================================================

print("=" * 80)
print("DECISION TREE CLASSIFIER")
print("=" * 80)

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dt_model.fit(x_train, y_train)
y_pred = dt_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f"Accuracy: {accuracy:.2f}")
print()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

# ============================================================================
# Feature Scaling and XGBoost Re-training
# ============================================================================

print("=" * 80)
print("XGBOOST WITH FEATURE SCALING")
print("=" * 80)

columns_to_be_scaled = ['Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment',
                        'max_recent_level_of_deliq', 'recent_level_of_deliq',
                        'time_since_recent_enq', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr']

for i in columns_to_be_scaled:
    column_data = df_encoded[i].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_column = scaler.fit_transform(column_data)
    df_encoded[i] = scaled_column

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis=1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

# ============================================================================
# Hyperparameter Tuning - XGBoost
# ============================================================================

print("=" * 80)
print("HYPERPARAMETER TUNING - XGBOOST")
print("=" * 80)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
accuracy = best_model.score(x_test, y_test)
print("Test Accuracy:", accuracy)