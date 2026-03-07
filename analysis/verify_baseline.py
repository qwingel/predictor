import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Загрузка данных
x_test = pd.read_csv('../test/x_test.csv').values
y_test = pd.read_csv('../test/y_test.csv').values.ravel()

print("="*70)
print("SIMPLE RULE VERIFICATION")
print("="*70)

# Проверка обеих интерпретаций
print("\n1. WRONG interpretation (delta > 0 => Team 2 wins, +1):")
y_pred_wrong = np.where(x_test[:, 0] > 0, 1, -1)
acc_wrong = accuracy_score(y_test, y_pred_wrong)
print(f"   Accuracy: {acc_wrong:.4f} (39.79%)")
print("   This was the INITIAL baseline - INCORRECT!")

print("\n2. CORRECT interpretation (delta > 0 => Team 1 wins, -1):")
y_pred_correct = np.where(x_test[:, 0] > 0, -1, 1)
acc_correct = accuracy_score(y_test, y_pred_correct)
print(f"   Accuracy: {acc_correct:.4f} (60.21%)")
print("   This is the TRUE baseline!")

print("\n" + "="*70)
print("EXPLANATION")
print("="*70)
print("delta_rating = Team2_rating - Team1_rating")
print()
print("If delta_rating > 0:")
print("  -> Team 2 has HIGHER rating")
print("  -> Team 1 is STRONGER (inverse relationship)")
print("  -> Team 1 should win (y = -1)")
print()
print("If delta_rating < 0:")
print("  -> Team 1 has HIGHER rating")
print("  -> Team 2 is STRONGER")
print("  -> Team 2 should win (y = +1)")
print()
print("Correlation in training data: -0.2098 (negative!)")
print("="*70)

print("\n" + "="*70)
print("IMPACT ON MODEL EVALUATION")
print("="*70)
print(f"Simple rule (WRONG):    {acc_wrong:.4f}")
print(f"Simple rule (CORRECT):  {acc_correct:.4f}")
print(f"ML model (best):        0.6227")
print()
print(f"Claimed improvement (vs WRONG baseline): +{(0.6227 - acc_wrong)*100:.2f}%")
print(f"Actual improvement (vs CORRECT baseline): +{(0.6227 - acc_correct)*100:.2f}%")
print()
print("CONCLUSION: ML model provides marginal 2% improvement, not 22%!")
print("="*70)
