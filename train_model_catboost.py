import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score
from catboost import CatBoostClassifier
import numpy as np

try:
    df = pd.read_csv("ctg-studies.csv")
except FileNotFoundError:
    print("Error: ctg-studies.csv not found. Please ensure the file is in the correct directory.")
    exit()

def preprocess_data(dataframe):
    df_filtered = dataframe.copy()

    if df_filtered.empty:
        print("Warning: No data found for training.")
        return pd.DataFrame(), pd.Series()
    df_filtered['target_outcome'] = df_filtered['Study Results'].apply(
        lambda result: 1 if result == 'YES' else 0
    )
    y_data = df_filtered['target_outcome']
    features = [
        'Study Type', 'Conditions', 'Interventions', 'Start Date',
        'Sponsor', 'Collaborators', 'Study Status', 'Study Title',
        'Primary Outcome Measures', 'Secondary Outcome Measures'
    ]
    X_data = df_filtered[features]
    return X_data, y_data

X, y = preprocess_data(df)

X = X.fillna('missing')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("=" * 60)
print("CLASS DISTRIBUTION ANALYSIS")
print("=" * 60)
print("\nTraining set distribution:")
print(y_train.value_counts())
print(f"Pass ratio: {(y_train == 1).sum() / len(y_train):.2%}")

print("\nTest set distribution:")
print(y_test.value_counts())
print(f"Pass ratio: {(y_test == 1).sum() / len(y_test):.2%}")

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count if pos_count != 0 else 1

print(f"\nScale Pos Weight: {scale_pos_weight:.2f}")

categorical_features = [
    'Study Type', 'Conditions', 'Interventions', 'Start Date',
    'Sponsor', 'Collaborators', 'Study Status', 'Study Title',
    'Primary Outcome Measures', 'Secondary Outcome Measures'
]

# Get categorical feature indices
cat_indices = [X_train.columns.get_loc(col) for col in categorical_features]

model = CatBoostClassifier(
    iterations=200,
    random_state=42,
    verbose=20,
    auto_class_weights='balanced',  # Automatically calculate weights based on class distribution
    eval_metric='AUC',  # Use AUC for imbalanced data instead of Accuracy
    task_type='CPU',
    cat_features=cat_indices  # Tell CatBoost which features are categorical
)

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# 5. Train the model
model.fit(
    X_train, 
    y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=20,
    plot=False
)

print("\nModel training complete!")

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"True Negatives (Correctly predicted Fail):  {cm[0,0]}")
print(f"False Positives (Fail mispredicted as Pass): {cm[0,1]}")
print(f"False Negatives (Pass mispredicted as Fail): {cm[1,0]}")
print(f"True Positives (Correctly predicted Pass):  {cm[1,1]}")

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {auc:.4f}")

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Top 10)")
print("=" * 60)
feature_importance = model.get_feature_importance()
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(importance_df.head(10))
