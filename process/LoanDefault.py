"""
Use Case : Train a random classifier model to classify a loan as default (1) or not default (0) based on
attributes like income, credit score and age
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Data
data = {
    'income': [40000, 50000, 60000, 80000, 30000, 70000, 120000, 25000],
    'credit_score': [600, 650, 700, 750, 580, 720, 800, 550],
    'age': [25, 30, 35, 40, 22, 38, 45, 20],
    'default': [1, 0, 0, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)
print(df)

# Features & target
X = df[['income', 'credit_score', 'age']]
y = df['default']

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("\nX_train:\n", X_train)
print("\ny_train:\n", y_train)
print("\nX_test:\n", X_test)
print("\ny_test:\n", y_test)
print("\ny_pred:\n", y_pred)

#Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending = False)
print("\nFeature Importances:\n")
print(importance_df)

# Precision — Think of it as “How accurate are my positive predictions?”
# When I predict someone will default, how often am I correct?
# Formula: Precision = True Positives / (True Positives + False Positives)
#
# Recall — Think of it as “How many of the real positives did I find?”
# Of all people who actually defaulted, how many did I correctly identify?
# Formula:  Recall = True Positives / (True Positives + False Negatives)
#
# F1-Score - Balance between Precision and Recall
# The F1-score combines them so we can compare models more fairly.
# F1 = 2 x ((Precision x Recall)/(Precision + Recall))



