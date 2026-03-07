import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Загрузка данных
x_train = pd.read_csv('../train/x_train.csv').values
x_test = pd.read_csv('../test/x_test.csv').values
y_train = pd.read_csv('../train/y_train.csv').values.ravel()
y_test = pd.read_csv('../test/y_test.csv').values.ravel()

print("="*60)
print("FINAL OPTIMIZED MODEL")
print("="*60)

# Оптимальная конфигурация из экспериментов
features_optimized = [0, 3, 4, 5, 6]  # delta_rating, map_missing_flag, LAN_flag, h2h_shrunk, h2h_count
best_kernel = "linear"
best_C = 0.5

print("\nModel Configuration:")
print(f"  Kernel: {best_kernel}")
print(f"  C: {best_C}")
print("\nFeatures used:")
print("  [0] delta_rating")
print("  [3] map_missing_flag")
print("  [4] LAN_flag")
print("  [5] h2h_shrunk")
print("  [6] h2h_count")

# Обучение финальной модели
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel=best_kernel, C=best_C))
])

final_model.fit(x_train[:, features_optimized], y_train)
y_pred = final_model.predict(x_test[:, features_optimized])

# Метрики
acc = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives:  {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives:  {cm[1,1]}")

# Сохранение модели
with open('../final_model.pkl', 'wb') as f:
    pickle.dump((final_model, features_optimized), f)
print(f"\n{'='*60}")
print("Model saved to: final_model.pkl")
print(f"{'='*60}")
