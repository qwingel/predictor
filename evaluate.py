"""
Модуль для оценки качества модели предсказания карт CS2.

Проверки:
1. Симметричность: pred(A, B) ≈ -pred(B, A)
2. ROC-AUC, Brier Score, Accuracy на тесте
3. Калибровочная кривая
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
import joblib
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, accuracy_score,
    ConfusionMatrixDisplay
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
import matplotlib
matplotlib.use('Agg')  # Для работы без дисплея
import matplotlib.pyplot as plt


def load_model(model_path: str = 'model_lgbm.pkl') -> lgb.LGBMClassifier:
    """
    Загружает модель из файла.
    """
    return joblib.load(model_path)


def load_test_data(features_path: str = 'features.csv') -> pd.DataFrame:
    """
    Загружает данные и возвращает тестовую выборку (последние ~15% по датам).
    Использует тот же метод сплита, что и train.py.
    """
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Сплит по уникальным датам (как в train.py)
    unique_dates = sorted(df['date'].unique())
    n_dates = len(unique_dates)
    val_date_idx = int(n_dates * 0.85)
    test_dates = unique_dates[val_date_idx:]

    test_df = df[df['date'].isin(test_dates)].copy()

    return test_df


def get_feature_columns() -> list:
    """
    Возвращает список колонок с признаками.
    """
    return [
        'delta_rating', 'form_diff_10', 'sos_form_diff_10',
        'days_since_last_match_diff', 'recent_matches_count_diff',
        'map_strength_diff', 'map_recent_form_diff',
        'map_t_side_winrate_diff', 'map_ct_side_winrate_diff', 'map_sample_diff',
        'rating_x_map_strength', 'form_x_map_form', 'rating_x_sos'
    ]


def prepare_data(df: pd.DataFrame, feature_cols: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Подготавливает данные для предсказания.
    """
    X = df[feature_cols].values
    # Преобразуем y из -1/1 в 0/1
    y_binary = (df['y'].values + 1) / 2
    return X, y_binary


def check_symmetry(
    model: lgb.LGBMClassifier,
    test_df: pd.DataFrame,
    n_samples: int = 100
) -> float:
    """
    Проверяет симметричность модели.

    Для случайных пар из тестовой выборки вычисляет предсказание для (A,B) и (B,A).
    Должно быть: pred_ab ≈ -pred_ba (или prob_ab ≈ 1 - prob_ba)

    Returns:
        mean_error: средняя ошибка симметричности
    """
    feature_cols = get_feature_columns()
    test_df = test_df.copy()

    errors = []

    # Берём случайные записи
    sampled = test_df.sample(n=min(n_samples, len(test_df)), random_state=42)

    for idx, row in sampled.iterrows():
        # Оригинальные признаки
        X_orig = row[feature_cols].values.reshape(1, -1)

        # Инвертированные признаки (для поменянных местами команд)
        # Все дельта-признаки инвертируются
        X_inv = -X_orig.copy()

        # map_name и is_lan остаются теми же (они одинаковы для обеих команд)
        # Но в наших признаках map_name не входит в feature_cols (категориальный)
        # is_lan тоже не входит (он одинаков)

        # Предсказания (вероятности класса 1 - победа team1)
        prob_orig = model.predict_proba(X_orig)[0, 1]
        prob_inv = model.predict_proba(X_inv)[0, 1]

        # Для симметричной модели: prob(A,B) + prob(B,A) = 1
        # Ошибка: |prob_orig + prob_inv - 1|
        error = abs(prob_orig + prob_inv - 1)
        errors.append(error)

    mean_error = np.mean(errors)

    print("\n" + "=" * 60)
    print("ПРОВЕРКА СИММЕТРИЧНОСТИ МОДЕЛИ")
    print("=" * 60)
    print(f"Проверено пар: {len(errors)}")
    print(f"Средняя ошибка симметричности: {mean_error:.6f}")
    print(f"  (идеал: 0, допустимо: < 0.05)")

    if mean_error < 0.05:
        print("✓ Модель симметрична")
    else:
        print("✗ Модель недостаточно симметрична")

    return mean_error


def compute_metrics(
    model: lgb.LGBMClassifier,
    test_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Вычисляет все метрики на тестовой выборке.
    """
    feature_cols = get_feature_columns()
    X_test, y_test = prepare_data(test_df, feature_cols)

    # Предсказания (вероятности)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Предсказания классов
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Метрики
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'roc_auc': roc_auc,
        'brier': brier,
        'accuracy': accuracy,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: str = 'calibration_curve.png',
    n_bins: int = 10
) -> None:
    """
    Строит калибровочную кривую.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Вычисляем калибровочную кривую
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )

    # Идеальная калибровка
    ax.plot([0, 1], [0, 1], "k--", label="Идеальная калибровка")

    # Фактическая калибровка
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Модель")

    ax.set_xlabel("Предсказанная вероятность (средняя по бину)")
    ax.set_ylabel("Доля положительных исходов")
    ax.set_title("Калибровочная кривая модели")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nКалибровочная кривая сохранена в {output_path}")


def plot_predictions_distribution(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    output_path: str = 'predictions_distribution.png'
) -> None:
    """
    Строит график распределения предсказаний.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Гистограмма предсказаний
    axes[0].hist(y_pred_proba, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0.5, color='r', linestyle='--', label='Порог 0.5')
    axes[0].set_xlabel('Предсказанная вероятность')
    axes[0].set_ylabel('Количество')
    axes[0].set_title('Распределение предсказаний')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Предсказания vs Фактические исходы
    axes[1].scatter(y_pred_proba, y_true, alpha=0.3, s=10)
    axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Предсказанная вероятность')
    axes[1].set_ylabel('Фактический исход (0/1)')
    axes[1].set_title('Предсказания vs Фактические исходы')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"График распределения сохранён в {output_path}")


def evaluate_full(
    model_path: str = 'model_lgbm.pkl',
    features_path: str = 'features.csv',
    report_path: str = 'evaluate_report.txt'
) -> Dict:
    """
    Полный пайплайн оценки модели.

    Returns:
        results: словарь с результатами
    """
    print("=" * 60)
    print("ОЦЕНКА МОДЕЛИ PREDICTION CS2")
    print("=" * 60)

    # Загружаем модель и данные
    print("\nЗагрузка модели...")
    model = load_model(model_path)
    print(f"Модель загружена из {model_path}")

    print("\nЗагрузка тестовых данных...")
    test_df = load_test_data(features_path)
    print(f"Тестовая выборка: {len(test_df)} записей")

    # Проверка симметричности
    symmetry_error = check_symmetry(model, test_df, n_samples=100)

    # Вычисление метрик
    print("\n" + "=" * 60)
    print("МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 60)

    metrics = compute_metrics(model, test_df)

    print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"  Brier Score:  {metrics['brier']:.4f}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")

    # Построение графиков
    print("\n" + "=" * 60)
    print("ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 60)

    plot_calibration_curve(
        metrics['y_test'],
        metrics['y_pred_proba'],
        output_path='calibration_curve.png'
    )

    plot_predictions_distribution(
        metrics['y_pred_proba'],
        metrics['y_test'],
        output_path='predictions_distribution.png'
    )

    # Сохранение отчёта
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ОТЧЁТ ОБ ОЦЕНКЕ МОДЕЛИ\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Дата оценки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("СИММЕТРИЧНОСТЬ:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Средняя ошибка: {symmetry_error:.6f}\n")
        if symmetry_error < 0.05:
            f.write("  Статус: ✓ Модель симметрична\n")
        else:
            f.write("  Статус: ✗ Модель недостаточно симметрична\n")

        f.write("\n" + "-" * 40 + "\n")
        f.write("МЕТРИКИ НА ТЕСТЕ:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  ROC-AUC:      {metrics['roc_auc']:.4f}\n")
        f.write(f"  Brier Score:  {metrics['brier']:.4f}\n")
        f.write(f"  Accuracy:     {metrics['accuracy']:.4f}\n")

        f.write("\n" + "-" * 40 + "\n")
        f.write("ЦЕЛЕВЫЕ ПОКАЗАТЕЛИ:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Целевой ROC-AUC: > 0.68\n")
        if metrics['roc_auc'] > 0.68:
            f.write(f"  Статус: ✓ ДОСТИГНУТ ({metrics['roc_auc']:.4f})\n")
        else:
            f.write(f"  Статус: ✗ НЕ ДОСТИГНУТ ({metrics['roc_auc']:.4f})\n")

        f.write("\n" + "-" * 40 + "\n")
        f.write("ГРАФИКИ:\n")
        f.write("-" * 40 + "\n")
        f.write("  - calibration_curve.png\n")
        f.write("  - predictions_distribution.png\n")

    print(f"\nОтчёт сохранён в {report_path}")

    results = {
        'symmetry_error': symmetry_error,
        'roc_auc': metrics['roc_auc'],
        'brier': metrics['brier'],
        'accuracy': metrics['accuracy']
    }

    return results


if __name__ == '__main__':
    results = evaluate_full()

    print("\n" + "=" * 60)
    print("ИТОГОВАЯ ОЦЕНКА")
    print("=" * 60)

    if results['roc_auc'] > 0.68:
        print(f"✓ Целевой ROC-AUC > 0.68 достигнут: {results['roc_auc']:.4f}")
    else:
        print(f"✗ Целевой ROC-AUC > 0.68 не достигнут: {results['roc_auc']:.4f}")

    if results['symmetry_error'] < 0.05:
        print(f"✓ Модель симметрична (ошибка: {results['symmetry_error']:.6f})")
    else:
        print(f"✗ Модель недостаточно симметрична (ошибка: {results['symmetry_error']:.6f})")
