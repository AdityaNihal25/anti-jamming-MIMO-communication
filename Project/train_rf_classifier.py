# train_rf_classifier.py
# Train a Random Forest classifier to predict jamming labels from 20k dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
csv_path = 'balanced_dataset_20000.csv'  # update path if needed
print("[INFO] Loading dataset...")
df = pd.read_csv(csv_path)

# Separate features and label
X = df.drop(columns=['label'])
y = df['label']

# Normalize features
print("[INFO] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
print("[INFO] Training Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n[INFO] Classification Report:")
print(classification_report(y_test, y_pred))

print("[INFO] Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and scaler
joblib.dump(clf, 'rf_model.pkl')
joblib.dump(scaler, 'rf_scaler.pkl')
print("[INFO] Model and scaler saved as 'rf_model.pkl' and 'rf_scaler.pkl'")
