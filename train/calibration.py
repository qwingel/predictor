# calibration.py
import numpy as np
from sklearn.calibration import calibration_curve
from scipy.optimize import minimize_scalar


def apply_temperature(probs, temperature):
    """
    Применяет температурное масштабирование к вероятностям.
    temperature > 1.0 — делает вероятности менее уверенными (ближе к 0.5)
    temperature < 1.0 — делает вероятности более уверенными
    """
    eps = 1e-7
    probs_clipped = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs_clipped / (1 - probs_clipped))
    scaled_logits = logits / temperature
    return 1 / (1 + np.exp(-scaled_logits))


def find_best_temperature(model, X_val, y_val, temperature_range=(0.5, 2.0)):
    """
    Находит оптимальный temperature, минимизирующий ECE.
    """
    raw_probs = model.predict_proba(X_val)[:, 1]

    def compute_ece(probs, y_true, n_bins=10):
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins)
        return np.mean(np.abs(prob_true - prob_pred))

    def objective(temp):
        calibrated = apply_temperature(raw_probs, temp)
        return compute_ece(calibrated, y_val)

    result = minimize_scalar(objective, bounds=temperature_range, method='bounded')
    return result.x, objective(1.0), result.fun


def calibrate_model(model, val_df, feature_cols):
    """
    Калибрует модель и сохраняет temperature в атрибут.
    """
    X_val = val_df[feature_cols].values
    y_val = (val_df['y'].values + 1) / 2

    optimal_temp, ece_before, ece_after = find_best_temperature(model, X_val, y_val)

    print(f"Temperature: {optimal_temp:.3f}")
    print(f"ECE до калибровки: {ece_before:.4f}")
    print(f"ECE после калибровки: {ece_after:.4f}")

    model.temperature_ = optimal_temp
    return model