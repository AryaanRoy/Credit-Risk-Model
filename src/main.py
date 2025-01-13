import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer

# Set basic style
plt.style.use('default')

# Load and analyze the dataset
print("Loading and analyzing data...")
data = pd.read_csv('data/credit_risk_dataset.csv')

# Check for missing values before preprocessing
print("\nMissing values before preprocessing:")
print(data.isnull().sum())

# Feature engineering (before handling missing values to avoid propagating NaNs)
print("\nPerforming feature engineering...")
# Creating copy to avoid SettingWithCopyWarning
data = data.copy()

# Creating new features
data['debt_to_income'] = data['loan_amnt'] / data['person_income']
data['is_high_interest'] = (data['loan_int_rate'] > data['loan_int_rate'].median()).astype(int)
data['age_to_employment_ratio'] = data['person_age'] / (data['person_emp_length'] + 1)

# Handling missing values
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Imputers
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Impute numeric columns
data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

# Impute categorical columns
data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

# Checking if there are any missing values left
print("\nMissing values after imputation:")
print(data.isnull().sum())

# Convert categorical variables
data = pd.get_dummies(data, drop_first=True)

# Separating features and target
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Visualization of class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=y)
plt.title('Class Distribution Before SMOTE')
plt.savefig('class_distribution.png')
plt.close()

# TRAINING THE MODELS
models = {
    'logistic': LogisticRegression(max_iter=2000, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingClassifier(random_state=42)
}

results = {}
plt.figure(figsize=(15, 5))

# Plot counter for ROC curves
plot_idx = 1

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Creating pipeline with proper scaling before SMOTE
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Cross-validation with error handling
    try:
        cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"Cross-validation ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    except Exception as e:
        print(f"Cross-validation failed: {str(e)}")
        continue
    
    # Fit model
    model_pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    # Storing results
    results[name] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'classification_report': classification_report(y_test, y_pred),
        'model': model_pipeline,
        'predictions': y_pred_proba
    }
    
    # Plot ROC curve for each model
    plt.subplot(1, 3, plot_idx)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC-AUC = {results[name]["roc_auc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC Curve')
    plt.legend()
    plot_idx += 1

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Find best model
best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
best_model = results[best_model_name]['model']
print(f"\nBest performing model: {best_model_name}")

# Feature importance plot for best model
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importances.head(10), x='importance', y='feature')
    plt.title(f'Top 10 Most Important Features - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Print final results
print("\nFinal Results:")
print("-" * 50)
for name, result in results.items():
    print(f"\n{name} Results:")
    print(f"ROC-AUC Score: {result['roc_auc']:.3f}")
    print("\nClassification Report:")
    print(result['classification_report'])

# Predict on the test set and identify risky applicants (loan_status = 1)
y_pred_risky = best_model.predict(X_test)
X_test['predicted_risk'] = y_pred_risky
X_test['actual_status'] = y_test.values

# Extract preticted risky applicants 
risky_applicants = X_test[X_test['predicted_risk'] == 1]
print("\nRisky Loan Applicants:")
print(risky_applicants)
