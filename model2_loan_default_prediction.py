"""
Loan Default Prediction
Model 2 — Binary Classification Pipeline
Dataset: loan_data.csv (Kaggle: udaymalviya/bank-loan-data)
Target: loan_status (0 = repaid, 1 = default)

Author: Angela Nguyen Hao
GitHub: https://github.com/angelaadida
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

sns.set(style='whitegrid')

# ─────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. LOAD DATA")
print("=" * 60)

df = pd.read_csv('loan_data.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nTarget distribution (loan_status):")
print(df['loan_status'].value_counts())
print(df['loan_status'].value_counts(normalize=True).round(4))

# ─────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. PREPROCESSING")
print("=" * 60)

# Drop leakage columns
# loan_int_rate      → priced based on risk AFTER the approval/default relationship is known
# loan_percent_income → = loan_amnt / person_income; loan_amnt itself is dropped below
# loan_amnt          → not needed as a predictor here; we keep the pipeline focused on
#                       pre-application attributes only (loan_amnt is a decision output,
#                       not an input available before deciding approval risk)
df = df.drop(columns=['loan_int_rate', 'loan_percent_income', 'loan_amnt'])
print(f"After dropping leakage columns: {df.shape}")

# Check missing values
missing = df.isnull().sum()
print(f"Missing values: {missing.sum()}")

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# One-hot encode categorical columns — DONE ONCE ONLY
cat_cols = ['person_gender', 'person_education', 'person_home_ownership',
            'loan_intent', 'previous_loan_defaults_on_file']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(f"After encoding: {df.shape}")

obj_remaining = df.select_dtypes(include='object').columns.tolist()
assert len(obj_remaining) == 0, f"Object columns remain: {obj_remaining}"
print("✅ All columns numeric")

# ─────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. FEATURE ENGINEERING")
print("=" * 60)

# Income per year of employment — financial stability proxy
df['Income_per_Emp_Year'] = df['person_income'] / (df['person_emp_exp'] + 1)

# Credit history density per year of age
df['Credit_History_per_Age'] = df['cb_person_cred_hist_length'] / (df['person_age'] + 1)

print("New features: Income_per_Emp_Year, Credit_History_per_Age")

# ─────────────────────────────────────────────────────────────────
# 4. TRAIN-TEST SPLIT & SCALING
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. TRAIN-TEST SPLIT & SCALING")
print("=" * 60)

X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Stratified split to preserve class ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
assert len(set(X_train.index) & set(X_test.index)) == 0
print("✅ No index overlap")
print(f"Train class balance:\n{y_train.value_counts(normalize=True).round(4)}")
print(f"Test class balance:\n{y_test.value_counts(normalize=True).round(4)}")

# Scale AFTER split — fit ONLY on X_train
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X.columns, index=X_test.index
)
print("✅ Scaling complete — fit on X_train only")

# ─────────────────────────────────────────────────────────────────
# 5. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. MODEL TRAINING & EVALUATION")
print("=" * 60)

def evaluate_model(y_true, y_pred, y_proba, model_name):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_proba)
    print(f"\n📊 {model_name}:")
    print(f"   Accuracy  : {acc:>8.4f}")
    print(f"   Precision : {prec:>8.4f}")
    print(f"   Recall    : {rec:>8.4f}")
    print(f"   F1-Score  : {f1:>8.4f}")
    print(f"   AUC-ROC   : {auc:>8.4f}")
    print("   Confusion Matrix:")
    print("   ", confusion_matrix(y_true, y_pred))
    print("-" * 45)
    return {"Model": model_name, "Accuracy": round(acc, 4), "Precision": round(prec, 4),
            "Recall": round(rec, 4), "F1": round(f1, 4), "AUC_ROC": round(auc, 4)}

# Model 1: Logistic Regression (baseline) — imbalanced data handled via class_weight
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# Model 2: Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
)
rf_clf.fit(X_train_scaled, y_train)
y_pred_rf = rf_clf.predict(X_test_scaled)
y_proba_rf = rf_clf.predict_proba(X_test_scaled)[:, 1]

# Model 3: XGBoost Classifier — scale_pos_weight handles imbalance
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
xgb_clf = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, random_state=42,
    eval_metric='logloss', n_jobs=-1
)
xgb_clf.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_clf.predict(X_test_scaled)
y_proba_xgb = xgb_clf.predict_proba(X_test_scaled)[:, 1]

results = []
results.append(evaluate_model(y_test, y_pred_lr, y_proba_lr, "Logistic Regression"))
results.append(evaluate_model(y_test, y_pred_rf, y_proba_rf, "Random Forest"))
results.append(evaluate_model(y_test, y_pred_xgb, y_proba_xgb, "XGBoost"))

results_df = pd.DataFrame(results).set_index("Model")
print("\n📊 Final Model Comparison:")
print(results_df.to_string())

# Pick best model by AUC-ROC for downstream feature importance / export
best_model_name = results_df['AUC_ROC'].idxmax()
print(f"\n🏆 Best model by AUC-ROC: {best_model_name}")

model_map = {"Logistic Regression": (log_reg, y_pred_lr, y_proba_lr),
             "Random Forest": (rf_clf, y_pred_rf, y_proba_rf),
             "XGBoost": (xgb_clf, y_pred_xgb, y_proba_xgb)}
best_model, y_pred_best, y_proba_best = model_map[best_model_name]

print("\nClassification Report (Best Model):")
print(classification_report(y_test, y_pred_best, target_names=['Repaid (0)', 'Default (1)']))

# 5-fold Stratified Cross-Validation on training set only (best model type)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    clone(best_model), X_train_scaled, y_train, cv=cv, scoring='roc_auc'
)
print(f"\nCross-Validation AUC-ROC ({best_model_name}, 5-fold): {cv_scores.mean():.4f} ±{cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (from the best model selected above)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. FEATURE IMPORTANCE")
print("=" * 60)

if hasattr(best_model, 'feature_importances_'):
    importance_values = best_model.feature_importances_
else:
    # Logistic Regression has no feature_importances_; use |coefficient| as a proxy
    importance_values = np.abs(best_model.coef_[0])

feature_importance = pd.Series(
    importance_values,
    index=X.columns
).sort_values(ascending=False)

print(f"Top 10 Features ({best_model_name}):")
print(feature_importance.head(10).round(4).to_string())

# ─────────────────────────────────────────────────────────────────
# 7. TABLEAU EXPORT — test rows only
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. TABLEAU EXPORT")
print("=" * 60)

df_raw = pd.read_csv('loan_data.csv')
df_export = df_raw.loc[X_test.index].copy()
df_export['Predicted_Loan_Status'] = y_pred_best
df_export['Default_Probability']  = y_proba_best.round(4)
df_export['Prediction_Correct']   = (df_export['loan_status'] == df_export['Predicted_Loan_Status'])
df_export['Split']                = 'Test'
df_export.to_csv('loan_default_tableau_export.csv', index=False)
print(f"✅ Exported: loan_default_tableau_export.csv ({len(df_export):,} rows)")

importance_df = pd.DataFrame({
    'Feature': feature_importance.index,
    'Importance_Score': feature_importance.values
})
importance_df.to_csv('feature_importance_default.csv', index=False)
print("✅ Exported: feature_importance_default.csv")

results_df.reset_index().to_csv('model2_metrics_summary.csv', index=False)
print("✅ Exported: model2_metrics_summary.csv")

# Confusion matrix as tidy CSV for Tableau
cm = confusion_matrix(y_test, y_pred_best)
cm_df = pd.DataFrame(
    cm, index=['Actual_Repaid', 'Actual_Default'],
    columns=['Predicted_Repaid', 'Predicted_Default']
).reset_index().rename(columns={'index': 'Actual'})
cm_df.to_csv('confusion_matrix_export.csv', index=False)
print("✅ Exported: confusion_matrix_export.csv")

# ROC curve points for Tableau
fpr, tpr, _ = roc_curve(y_test, y_proba_best)
roc_df = pd.DataFrame({'False_Positive_Rate': fpr, 'True_Positive_Rate': tpr})
roc_df.to_csv('roc_curve_export.csv', index=False)
print("✅ Exported: roc_curve_export.csv")

# ─────────────────────────────────────────────────────────────────
# 8. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. SAVING VISUALIZATIONS")
print("=" * 60)

# Model comparison chart
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
colors = ['tomato', 'steelblue', 'seagreen']
for ax, metric in zip(axes, ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC_ROC']):
    vals = results_df[metric]
    bars = ax.bar(vals.index, vals.values, color=colors[:len(vals)], edgecolor='black', width=0.5)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.3f}',
                ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=25)
plt.suptitle('Model Comparison — Test Set', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('model2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: model2_comparison.png")

# Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Repaid', 'Default'], yticklabels=['Repaid', 'Default'])
plt.title(f'Confusion Matrix — {best_model_name}', fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: confusion_matrix.png")

# ROC curve
plt.figure(figsize=(7, 6))
for name, (_, _, proba) in model_map.items():
    fpr_i, tpr_i, _ = roc_curve(y_test, proba)
    auc_i = roc_auc_score(y_test, proba)
    plt.plot(fpr_i, tpr_i, label=f'{name} (AUC={auc_i:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — All Models', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: roc_curve.png")

# Feature importance
plt.figure(figsize=(12, 7))
feature_importance.head(15).plot(kind='bar', color='steelblue', edgecolor='black')
plt.title(f'Top 15 Feature Importance — {best_model_name}', fontsize=13, fontweight='bold')
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance_default.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: feature_importance_default.png")

print("\n" + "=" * 60)
print("✅ ALL DONE")
print("=" * 60)
