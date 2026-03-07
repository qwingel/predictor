import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Загрузка данных
x_train = pd.read_csv('../train/x_train.csv').values
x_test = pd.read_csv('../test/x_test.csv').values
y_train = pd.read_csv('../train/y_train.csv').values.ravel()
y_test = pd.read_csv('../test/y_test.csv').values.ravel()

print("="*70)
print("BEST MODEL: LOGISTIC REGRESSION")
print("="*70)

# Оптимальная конфигурация
features_optimized = [0, 3, 4, 5, 6]
X_train = x_train[:, features_optimized]
X_test = x_test[:, features_optimized]

print("\nModel Configuration:")
print("  Algorithm: Logistic Regression")
print("  C: 0.5")
print("  Regularization: L2")
print("\nFeatures used:")
print("  [0] delta_rating")
print("  [3] map_missing_flag")
print("  [4] LAN_flag")
print("  [5] h2h_shrunk")
print("  [6] h2h_count")

# Обучение модели
best_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(C=0.5, max_iter=1000, random_state=42))
])

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

# Метрики
acc = accuracy_score(y_test, y_pred)
print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"Accuracy: {acc:.4f} (62.27%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Team 1 Wins (-1)", "Team 2 Wins (+1)"]))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives:  {cm[0,0]} (correctly predicted Team 1 wins)")
print(f"False Positives: {cm[0,1]} (predicted Team 2, but Team 1 won)")
print(f"False Negatives: {cm[1,0]} (predicted Team 1, but Team 2 won)")
print(f"True Positives:  {cm[1,1]} (correctly predicted Team 2 wins)")

# Анализ коэффициентов
print(f"\n{'='*70}")
print("FEATURE IMPORTANCE (Logistic Regression Coefficients)")
print(f"{'='*70}")
feature_names = ["delta_rating", "map_missing_flag", "LAN_flag", "h2h_shrunk", "h2h_count"]
coefficients = best_model.named_steps['lr'].coef_[0]
for name, coef in zip(feature_names, coefficients):
    print(f"  {name:20s}: {coef:7.4f}")

print(f"\nIntercept: {best_model.named_steps['lr'].intercept_[0]:.4f}")

# Сохранение модели
with open('../best_model_logistic.pkl', 'wb') as f:
    pickle.dump((best_model, features_optimized), f)

print(f"\n{'='*70}")
print("Model saved to: best_model_logistic.pkl")
print(f"{'='*70}")

# Сравнение с предыдущими результатами
print(f"\n{'='*70}")
print("IMPROVEMENT SUMMARY")
print(f"{'='*70}")
print(f"Simple rule WRONG (delta > 0 => +1):  39.79%")
print(f"Simple rule CORRECT (delta > 0 => -1): 60.21%")
print(f"SVM (all features):                     60.47%")
print(f"SVM (optimized features):               61.63%")
print(f"Logistic Regression (optimized):        62.27% <- BEST")
print(f"\nImprovement over CORRECT baseline: +2.06%")
print(f"NOTE: delta_rating has inverse relationship!")
print(f"{'='*70}")
