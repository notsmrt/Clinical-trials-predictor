import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



try:
    df = pd.read_csv("all-studies.csv")
except FileNotFoundError:
    print("Error: all-studies.csv not found. Please ensure the file is in the correct directory.")
    exit()

def preprocess_data(dataframe):
    # trgt variable study results
    df_filtered = dataframe.copy() 

    if df_filtered.empty:
        print("Warning: No data found for training.")
        return pd.DataFrame(), pd.Series()

    # Map to numerical target
    df_filtered['target_outcome'] = df_filtered['Study Results'].apply(
        lambda result: 1 if result == 'YES' else 0
    )
    y_data = df_filtered['target_outcome']

    df_filtered['Enrollment'] = pd.to_numeric(df_filtered.get('Enrollment', 0), errors='coerce').fillna(0)
    df_filtered['Start Year'] = pd.to_datetime(df_filtered.get('Start Date', ''), errors='coerce').dt.year.fillna(0).astype(int)

    df_filtered['Condition Count'] = df_filtered.get('Conditions', '').fillna('').astype(str).apply(lambda x: len([v for v in x.split('|') if v.strip()]))
    df_filtered['Intervention Count'] = df_filtered.get('Interventions', '').fillna('').astype(str).apply(lambda x: len([v for v in x.split('|') if v.strip()]))

    features = [
        'Study Title',
        'Conditions',
        'Interventions',
        'Sponsor',
        'Collaborators',
        'Study Type',
        'Study Design',
        'Sex',
        'Age',
        'Phases',
        'Funder Type',
        'Enrollment',
        'Start Date',
        'Start Year',
        'Condition Count',
        'Intervention Count'
    ]

    X_data = df_filtered[features].copy()
    return X_data, y_data

X, y = preprocess_data(df)
print("Data preprocessing complete.")


categorical_features = [
    "Study Title",
    "Conditions",
    "Interventions",
    "Sponsor",
    "Collaborators",
    "Study Type",
    "Study Design",
    "Sex",
    "Age",
    "Phases",
    "Funder Type",
    "Start Date"
]

# Numeric features distilled from dataset
numeric_features = [
    'Enrollment',
    'Start Year',
    'Condition Count',
    'Intervention Count'
]

# Create transformers
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create a preprocessor 
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)
    ],
    remainder='drop'
)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# pipeline with weights to adjust conservative bias
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(random_state=42, class_weight={0: 1, 1: 1}, max_iter=1000))])

print("Starting model training...")
model.fit(X_train, y_train)
print("Model training complete.")

y_pred = model.predict(X_test)

# classification report
print("Model Evaluation Results:\n")
# update to fail or pass for trgt name
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))
cm = confusion_matrix(y_test, y_pred)

# Executive summary: four separate confusion matrix values
tn, fp, fn, tp = cm.ravel()
print(f"True negatives={tn}")
print(f"False negatives={fn}")
print("")
print(f"True positives={tp}")
print(f"False positives={fp}")