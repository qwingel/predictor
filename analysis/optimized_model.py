import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Загрузка данных
x_train = pd.read_csv('../train/x_train.csv').values
x_test = pd.read_csv('../test/x_test.csv').values
y_train = pd.read_csv('../train/y_train.csv').values.ravel()
y_test = pd.read_csv('../test/y_test.csv').values.ravel()

print("="*60)
print("OPTIMIZED MODEL (based on ablation study)")
print("="*60)

# На основе ablation study:
# - map_id ухудшает результат (убираем)
# - delta_map_wr_windowed не добавляет ничего (убираем)
# - h2h признаки полезны (оставляем)
# Используем: delta_rating (0), map_missing_flag (3), LAN_flag (4), h2h_shrunk (5), h2h_count (6)

features_optimized = [0, 3, 4, 5, 6]

print("\nFeatures used:")
print("  - delta_rating")
print("  - map_missing_flag")
print("  - LAN_flag")
print("  - h2h_shrunk")
print("  - h2h_count")
print("\nFeatures removed (harmful/useless):")
print("  - map_id (decreased accuracy by 0.52%)")
print("  - delta_map_wr_windowed (no impact)")

model_optimized = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="linear", C=1.0))
])

model_optimized.fit(x_train[:, features_optimized], y_train)
y_pred = model_optimized.predict(x_test[:, features_optimized])

acc = accuracy_score(y_test, y_pred)
print(f"\nOptimized Model Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
