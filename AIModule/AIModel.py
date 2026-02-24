import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load ALL datasets
files = [
    "dataset1.csv",
    "dataset2.csv",
    "dataset3.csv"
]

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)


# 2. Extract grade columns
grade_cols = [c for c in df.columns if c.startswith("grade_")]


# 3. Build training samples
X = []
y = []

for _, row in df.iterrows():
    grades = row[grade_cols].values
    grades = grades[~pd.isna(grades)]

    for i in range(1, len(grades)):
        prev = grades[i - 1]
        delta = prev - grades[i - 2] if i > 1 else 0
        X.append([prev, delta])
        y.append(grades[i])

X = np.array(X)
y = np.array(y)

print("Training samples:", len(X))


# 4. Model evaluation (K-Fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []
rmse_scores = []
r2_scores = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae_scores.append(mean_absolute_error(y_test, preds))
    rmse_scores.append(mean_squared_error(y_test, preds))
    r2_scores.append(r2_score(y_test, preds))

print("\nModel evaluation (5-fold CV):")
print("MAE:", round(np.mean(mae_scores), 3))
print("RMSE:", round(np.mean(rmse_scores), 3))
print("R²:", round(np.mean(r2_scores), 3))


# 5. Train final model on ALL data
final_model = LinearRegression()
final_model.fit(X, y)


# 6. Prediction function
def predict_next_grade(history):
    if len(history) == 1:
        prev = history[-1]
        delta = 0
    else:
        prev = history[-1]
        delta = history[-1] - history[-2]

    return final_model.predict([[prev, delta]])[0]


# 7. Example predictions
print("\nExample predictions:")

print("Student A (3.4 → ?):",
      round(predict_next_grade([3.4]), 2))

print("Student B (3.0 → 4.0 → ?):",
      round(predict_next_grade([3.0, 4.0]), 2))

print("Student C:",
      round(predict_next_grade([3.4, 3.7, 3.2, 2.5, 4.1]), 2))
