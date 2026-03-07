import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
x_train = pd.read_csv('../train/x_train.csv').values
x_test = pd.read_csv('../test/x_test.csv').values
y_train = pd.read_csv('../train/y_train.csv').values.ravel()
y_test = pd.read_csv('../test/y_test.csv').values.ravel()

# Оптимальные признаки из ablation study
features_optimized = [0, 3, 4, 5, 6]
X_train = x_train[:, features_optimized]
X_test = x_test[:, features_optimized]

print("="*70)
print("MODEL COMPARISON")
print("="*70)
print("\nFeatures: delta_rating, map_missing_flag, LAN_flag, h2h_shrunk, h2h_count")
print("\n" + "="*70)

results = []

# 1. Logistic Regression
print("\n1. Logistic Regression")
for C in [0.1, 0.5, 1.0, 2.0]:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append(("LogisticRegression", f"C={C}", acc))
    print(f"   C={C:3.1f} -> Accuracy: {acc:.4f}")

# 2. SVM (baseline from previous experiments)
print("\n2. SVM")
for kernel in ["linear", "rbf"]:
    for C in [0.5, 1.0, 2.0]:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=kernel, C=C, random_state=42))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append(("SVM", f"{kernel}, C={C}", acc))
        print(f"   {kernel:6s}, C={C:3.1f} -> Accuracy: {acc:.4f}")

# 3. Random Forest
print("\n3. Random Forest")
for n_est in [50, 100, 200]:
    for max_depth in [5, 10, None]:
        model = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        depth_str = str(max_depth) if max_depth else "None"
        results.append(("RandomForest", f"n={n_est}, depth={depth_str}", acc))
        print(f"   n_estimators={n_est:3d}, max_depth={depth_str:4s} -> Accuracy: {acc:.4f}")

# 4. Gradient Boosting
print("\n4. Gradient Boosting")
for n_est in [50, 100, 200]:
    for lr in [0.05, 0.1, 0.2]:
        model = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append(("GradientBoosting", f"n={n_est}, lr={lr}", acc))
        print(f"   n_estimators={n_est:3d}, lr={lr:4.2f} -> Accuracy: {acc:.4f}")

# Находим лучшую модель
print("\n" + "="*70)
print("TOP 5 MODELS")
print("="*70)
results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
for i, (model_name, params, acc) in enumerate(results_sorted[:5], 1):
    print(f"{i}. {model_name:20s} ({params:30s}) -> {acc:.4f}")

# Обучаем лучшую модель и показываем детали
best_model_name, best_params, best_acc = results_sorted[0]
print("\n" + "="*70)
print(f"BEST MODEL: {best_model_name} ({best_params})")
print(f"Accuracy: {best_acc:.4f}")
print("="*70)
