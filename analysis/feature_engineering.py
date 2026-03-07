import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
x_train = pd.read_csv('../train/x_train.csv').values
x_test = pd.read_csv('../test/x_test.csv').values
y_train = pd.read_csv('../train/y_train.csv').values.ravel()
y_test = pd.read_csv('../test/y_test.csv').values.ravel()

# Оптимальные признаки
features_optimized = [0, 3, 4, 5, 6]
X_train = x_train[:, features_optimized]
X_test = x_test[:, features_optimized]

print("="*70)
print("FEATURE ENGINEERING EXPERIMENTS")
print("="*70)

results = []

# Baseline: текущая лучшая модель
print("\n1. BASELINE (no feature engineering)")
model_baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(C=0.5, max_iter=1000, random_state=42))
])
model_baseline.fit(X_train, y_train)
y_pred = model_baseline.predict(X_test)
acc_baseline = accuracy_score(y_test, y_pred)
results.append(("Baseline", acc_baseline))
print(f"   Accuracy: {acc_baseline:.4f}")

# Эксперимент 2: Polynomial features (degree 2)
print("\n2. POLYNOMIAL FEATURES (degree=2)")
for C in [0.1, 0.5, 1.0]:
    model_poly = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000, random_state=42))
    ])
    model_poly.fit(X_train, y_train)
    y_pred = model_poly.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((f"Poly(2), C={C}", acc))
    print(f"   C={C:3.1f} -> Accuracy: {acc:.4f}")

# Эксперимент 3: Manual interaction features
print("\n3. MANUAL INTERACTION FEATURES")
# delta_rating * h2h_shrunk, delta_rating^2
X_train_interact = np.column_stack([
    X_train,
    X_train[:, 0] * X_train[:, 3],  # delta_rating * h2h_shrunk
    X_train[:, 0] ** 2,              # delta_rating^2
])
X_test_interact = np.column_stack([
    X_test,
    X_test[:, 0] * X_test[:, 3],
    X_test[:, 0] ** 2,
])
for C in [0.1, 0.5, 1.0]:
    model_interact = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000, random_state=42))
    ])
    model_interact.fit(X_train_interact, y_train)
    y_pred = model_interact.predict(X_test_interact)
    acc = accuracy_score(y_test, y_pred)
    results.append((f"Interact, C={C}", acc))
    print(f"   C={C:3.1f} -> Accuracy: {acc:.4f}")

# Эксперимент 4: Log transform of absolute delta_rating
print("\n4. LOG TRANSFORM (log(|delta_rating| + 1))")
X_train_log = X_train.copy()
X_test_log = X_test.copy()
X_train_log[:, 0] = np.sign(X_train[:, 0]) * np.log1p(np.abs(X_train[:, 0]))
X_test_log[:, 0] = np.sign(X_test[:, 0]) * np.log1p(np.abs(X_test[:, 0]))
for C in [0.1, 0.5, 1.0]:
    model_log = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000, random_state=42))
    ])
    model_log.fit(X_train_log, y_train)
    y_pred = model_log.predict(X_test_log)
    acc = accuracy_score(y_test, y_pred)
    results.append((f"Log, C={C}", acc))
    print(f"   C={C:3.1f} -> Accuracy: {acc:.4f}")

# Summary
print("\n" + "="*70)
print("SUMMARY - TOP 5 CONFIGURATIONS")
print("="*70)
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
for i, (name, acc) in enumerate(results_sorted[:5], 1):
    improvement = (acc - acc_baseline) * 100
    print(f"{i}. {name:30s} -> {acc:.4f} ({improvement:+.2f}%)")

print("\n" + "="*70)
if results_sorted[0][1] > acc_baseline:
    print(f"BEST: {results_sorted[0][0]} with {results_sorted[0][1]:.4f} accuracy")
    print(f"Improvement over baseline: +{(results_sorted[0][1] - acc_baseline)*100:.2f}%")
else:
    print("No improvement over baseline. Stick with original features.")
print("="*70)
