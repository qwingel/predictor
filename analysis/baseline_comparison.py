import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Загрузка данных
x_train = pd.read_csv('../train/x_train.csv').values
x_test = pd.read_csv('../test/x_test.csv').values
y_train = pd.read_csv('../train/y_train.csv').values.ravel()
y_test = pd.read_csv('../test/y_test.csv').values.ravel()

print("="*60)
print("BASELINE COMPARISON")
print("="*60)

# Baseline 1: Всегда выбирать команду с лучшим delta_rating
print("\n1. Simple Rule: Choose team with better delta_rating")
print("   WRONG rule (delta > 0 => +1):")
y_pred_wrong = np.where(x_test[:, 0] > 0, 1, -1)
acc_wrong = accuracy_score(y_test, y_pred_wrong)
print(f"      Accuracy: {acc_wrong:.4f}")

print("   CORRECT rule (delta > 0 => -1, i.e., Team 1 wins):")
y_pred_simple = np.where(x_test[:, 0] > 0, -1, 1)
acc_simple = accuracy_score(y_test, y_pred_simple)
print(f"      Accuracy: {acc_simple:.4f}")
print("   NOTE: Inverse relationship! Positive delta_rating means Team 1 is stronger")

# Baseline 2: Модель только на delta_rating
print("\n2. Model with ONLY delta_rating")
model_delta = SVC(kernel="linear", C=1.0)
model_delta.fit(x_train[:, [0]], y_train)
y_pred_delta = model_delta.predict(x_test[:, [0]])
acc_delta = accuracy_score(y_test, y_pred_delta)
print(f"   Accuracy: {acc_delta:.4f}")

# Baseline 3: Модель без h2h признаков (без индексов 5, 6)
print("\n3. Model WITHOUT h2h features (h2h_shrunk, h2h_count)")
features_no_h2h = [0, 1, 2, 3, 4]
cat_features_no_h2h = [1]
num_features_no_h2h = [0, 2, 3, 4]
preprocessor_no_h2h = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features_no_h2h),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features_no_h2h),
    ]
)
model_no_h2h = Pipeline([
    ("preprocessor", preprocessor_no_h2h),
    ("svc", SVC(kernel="linear", C=1.0))
])
model_no_h2h.fit(x_train[:, features_no_h2h], y_train)
y_pred_no_h2h = model_no_h2h.predict(x_test[:, features_no_h2h])
acc_no_h2h = accuracy_score(y_test, y_pred_no_h2h)
print(f"   Accuracy: {acc_no_h2h:.4f}")

# Baseline 4: Модель без map_id (без индекса 1)
print("\n4. Model WITHOUT map_id")
features_no_map = [0, 2, 3, 4, 5, 6]
preprocessor_no_map = StandardScaler()
model_no_map = Pipeline([
    ("preprocessor", preprocessor_no_map),
    ("svc", SVC(kernel="linear", C=1.0))
])
model_no_map.fit(x_train[:, features_no_map], y_train)
y_pred_no_map = model_no_map.predict(x_test[:, features_no_map])
acc_no_map = accuracy_score(y_test, y_pred_no_map)
print(f"   Accuracy: {acc_no_map:.4f}")

# Baseline 5: Модель без delta_map_wr_windowed (без индекса 2)
print("\n5. Model WITHOUT delta_map_wr_windowed")
features_no_mapwr = [0, 1, 3, 4, 5, 6]
cat_features_no_mapwr = [1]
num_features_no_mapwr = [0, 2, 3, 4, 5]
preprocessor_no_mapwr = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features_no_mapwr),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features_no_mapwr),
    ]
)
model_no_mapwr = Pipeline([
    ("preprocessor", preprocessor_no_mapwr),
    ("svc", SVC(kernel="linear", C=1.0))
])
model_no_mapwr.fit(x_train[:, features_no_mapwr], y_train)
y_pred_no_mapwr = model_no_mapwr.predict(x_test[:, features_no_mapwr])
acc_no_mapwr = accuracy_score(y_test, y_pred_no_mapwr)
print(f"   Accuracy: {acc_no_mapwr:.4f}")

# Full model: все признаки
print("\n6. FULL Model (all features)")
cat_features = [1]
num_features = [0, 2, 3, 4, 5, 6]
preprocessor_full = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)
model_full = Pipeline([
    ("preprocessor", preprocessor_full),
    ("svc", SVC(kernel="linear", C=1.0))
])
model_full.fit(x_train, y_train)
y_pred_full = model_full.predict(x_test)
acc_full = accuracy_score(y_test, y_pred_full)
print(f"   Accuracy: {acc_full:.4f}")

# Итоговая таблица
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Model':<45} {'Accuracy':>10}")
print("-"*60)
print(f"{'1. Simple rule (CORRECTED: delta > 0 => -1)':<45} {acc_simple:>10.4f}")
print(f"{'2. Only delta_rating':<45} {acc_delta:>10.4f}")
print(f"{'3. Without h2h features':<45} {acc_no_h2h:>10.4f}")
print(f"{'4. Without map_id':<45} {acc_no_map:>10.4f}")
print(f"{'5. Without delta_map_wr_windowed':<45} {acc_no_mapwr:>10.4f}")
print(f"{'6. FULL model (all features)':<45} {acc_full:>10.4f}")
print("="*60)

# Анализ вклада признаков
print("\nFEATURE CONTRIBUTION ANALYSIS:")
print(f"  Adding features to delta_rating: +{(acc_full - acc_delta)*100:.2f}%")
print(f"  Impact of removing h2h: {(acc_full - acc_no_h2h)*100:.2f}%")
print(f"  Impact of removing map_id: {(acc_full - acc_no_map)*100:.2f}%")
print(f"  Impact of removing delta_map_wr: {(acc_full - acc_no_mapwr)*100:.2f}%")
