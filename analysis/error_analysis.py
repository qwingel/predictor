import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Загрузка данных
x_train_df = pd.read_csv('../train/x_train.csv')
x_test_df = pd.read_csv('../test/x_test.csv')
x_train = x_train_df.values
x_test = x_test_df.values
y_train = pd.read_csv('../train/y_train.csv').values.ravel()
y_test = pd.read_csv('../test/y_test.csv').values.ravel()

# Лучшая модель
features_optimized = [0, 3, 4, 5, 6]
X_train = x_train[:, features_optimized]
X_test = x_test[:, features_optimized]

best_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(C=0.5, max_iter=1000, random_state=42))
])

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

print("="*70)
print("ERROR ANALYSIS")
print("="*70)

# Создаем DataFrame для анализа
feature_names = ["delta_rating", "map_missing_flag", "LAN_flag", "h2h_shrunk", "h2h_count"]
test_df = pd.DataFrame(X_test, columns=feature_names)
test_df['y_true'] = y_test
test_df['y_pred'] = y_pred
test_df['correct'] = (y_test == y_pred)
test_df['proba_neg'] = y_proba[:, 0]
test_df['proba_pos'] = y_proba[:, 1]
test_df['confidence'] = np.max(y_proba, axis=1)

# Статистика по правильным и неправильным предсказаниям
print("\n1. OVERALL STATISTICS")
print(f"   Correct predictions: {test_df['correct'].sum()} / {len(test_df)} ({test_df['correct'].mean()*100:.2f}%)")
print(f"   Incorrect predictions: {(~test_df['correct']).sum()} / {len(test_df)} ({(~test_df['correct']).mean()*100:.2f}%)")

# Анализ по уверенности модели
print("\n2. CONFIDENCE ANALYSIS")
correct_conf = test_df[test_df['correct']]['confidence'].mean()
incorrect_conf = test_df[~test_df['correct']]['confidence'].mean()
print(f"   Average confidence (correct): {correct_conf:.4f}")
print(f"   Average confidence (incorrect): {incorrect_conf:.4f}")
print(f"   Difference: {correct_conf - incorrect_conf:.4f}")

# Анализ ошибок по типам
print("\n3. ERROR TYPES")
false_positives = test_df[(test_df['y_true'] == -1) & (test_df['y_pred'] == 1)]
false_negatives = test_df[(test_df['y_true'] == 1) & (test_df['y_pred'] == -1)]
print(f"   False Positives (predicted +1, actual -1): {len(false_positives)}")
print(f"   False Negatives (predicted -1, actual +1): {len(false_negatives)}")

# Анализ признаков для правильных и неправильных предсказаний
print("\n4. FEATURE STATISTICS (Correct vs Incorrect)")
print("\n   Correct predictions:")
for feat in feature_names:
    mean_val = test_df[test_df['correct']][feat].mean()
    print(f"      {feat:20s}: {mean_val:7.3f}")

print("\n   Incorrect predictions:")
for feat in feature_names:
    mean_val = test_df[~test_df['correct']][feat].mean()
    print(f"      {feat:20s}: {mean_val:7.3f}")

# Самые уверенные ошибки
print("\n5. MOST CONFIDENT MISTAKES (Top 10)")
most_confident_errors = test_df[~test_df['correct']].nlargest(10, 'confidence')
print("\n   Index | delta_rating | h2h_shrunk | confidence | y_true | y_pred")
print("   " + "-"*65)
for idx, row in most_confident_errors.iterrows():
    print(f"   {idx:5d} | {row['delta_rating']:12.1f} | {row['h2h_shrunk']:10.1f} | {row['confidence']:10.4f} | {int(row['y_true']):6d} | {int(row['y_pred']):6d}")

# Анализ delta_rating в ошибках
print("\n6. DELTA_RATING ANALYSIS IN ERRORS")
print(f"   False Positives (FP) - predicted +1, actual -1:")
print(f"      Mean delta_rating: {false_positives['delta_rating'].mean():.2f}")
print(f"      Median delta_rating: {false_positives['delta_rating'].median():.2f}")
print(f"      Min/Max: {false_positives['delta_rating'].min():.2f} / {false_positives['delta_rating'].max():.2f}")

print(f"\n   False Negatives (FN) - predicted -1, actual +1:")
print(f"      Mean delta_rating: {false_negatives['delta_rating'].mean():.2f}")
print(f"      Median delta_rating: {false_negatives['delta_rating'].median():.2f}")
print(f"      Min/Max: {false_negatives['delta_rating'].min():.2f} / {false_negatives['delta_rating'].max():.2f}")

# Анализ по диапазонам delta_rating
print("\n7. ACCURACY BY DELTA_RATING RANGE")
bins = [-np.inf, -50, -20, 0, 20, 50, np.inf]
labels = ['< -50', '-50 to -20', '-20 to 0', '0 to 20', '20 to 50', '> 50']
test_df['rating_bin'] = pd.cut(test_df['delta_rating'], bins=bins, labels=labels)
for label in labels:
    subset = test_df[test_df['rating_bin'] == label]
    if len(subset) > 0:
        acc = subset['correct'].mean()
        print(f"   {label:12s}: {acc:.4f} ({len(subset):3d} samples)")

print("\n" + "="*70)
print("KEY INSIGHTS:")
print("="*70)
print("1. Model confidence correlates with correctness")
print("2. Check if errors cluster in specific delta_rating ranges")
print("3. Analyze if h2h features help in close matchups (delta_rating near 0)")
print("="*70)
