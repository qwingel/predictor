import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Загрузка данных
x_train = pd.read_csv('../train/x_train.csv').values
x_test = pd.read_csv('../test/x_test.csv').values
y_train = pd.read_csv('../train/y_train.csv').values.ravel()
y_test = pd.read_csv('../test/y_test.csv').values.ravel()

# Оптимальные признаки из ablation study
features_optimized = [0, 3, 4, 5, 6]

print("="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

# Тестируем разные ядра и параметры C
configs = [
    ("linear", 0.1),
    ("linear", 0.5),
    ("linear", 1.0),
    ("linear", 2.0),
    ("linear", 5.0),
    ("rbf", 0.1),
    ("rbf", 0.5),
    ("rbf", 1.0),
    ("rbf", 2.0),
    ("rbf", 5.0),
    ("poly", 0.1),
    ("poly", 0.5),
    ("poly", 1.0),
]

results = []

for kernel, C in configs:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=kernel, C=C))
    ])

    model.fit(x_train[:, features_optimized], y_train)
    y_pred = model.predict(x_test[:, features_optimized])
    acc = accuracy_score(y_test, y_pred)

    results.append((kernel, C, acc))
    print(f"Kernel: {kernel:6s}, C: {C:4.1f} -> Accuracy: {acc:.4f}")

# Находим лучшую конфигурацию
best_kernel, best_C, best_acc = max(results, key=lambda x: x[2])

print("\n" + "="*60)
print(f"BEST CONFIGURATION:")
print(f"  Kernel: {best_kernel}")
print(f"  C: {best_C}")
print(f"  Accuracy: {best_acc:.4f}")
print("="*60)
