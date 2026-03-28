"""
Главный модуль для запуска полного пайплайна предсказания карт CS2.

Запуск:
    python main.py

Пайплайн:
1. Построение признаков (features.py)
2. Обучение модели (train.py)
3. Оценка модели (evaluate.py)
"""
import sys
from datetime import datetime


def main():
    """
    Запускает полный пайплайн обучения и оценки модели.
    """
    print("=" * 70)
    print("  СИСТЕМА ПРЕДСКАЗАНИЯ ИСХОДА КАРТ CS2")
    print(f"  Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Шаг 1: Построение признаков
    print("\n" + "=" * 70)
    print("  ШАГ 1: ПОСТРОЕНИЕ ПРИЗНАКОВ")
    print("=" * 70)

    from features import build_features

    features_df = build_features(
        db_path='cs2_data.db',
        ratings_path='top_teams.txt',
        output_path='features.csv',
        verbose=True
    )

    print(f"\n[OK] Postroeno priznakov: {len(features_df)} zapisey")

    # Шаг 2: Обучение модели
    print("\n" + "=" * 70)
    print("  ШАГ 2: ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 70)

    from train import train_and_save

    model, test_metrics = train_and_save(
        features_path='features.csv',
        model_path='model_lgbm.pkl',
        report_path='train_report.txt'
    )

    print(f"\n[OK] Model obuchena, ROC-AUC na teste: {test_metrics['roc_auc']:.4f}")

    # Шаг 3: Оценка модели
    print("\n" + "=" * 70)
    print("  ШАГ 3: ОЦЕНКА МОДЕЛИ")
    print("=" * 70)

    from evaluate import evaluate_full

    eval_results = evaluate_full(
        model_path='model_lgbm.pkl',
        features_path='features.csv',
        report_path='evaluate_report.txt'
    )

    # Итоговый вывод
    print("\n" + "=" * 70)
    print("  ITOGOVYY OTCHYOT")
    print("=" * 70)

    print(f"\n  Метрики на тесте:")
    print(f"    ROC-AUC:       {eval_results['roc_auc']:.4f}")
    print(f"    Brier Score:   {eval_results['brier']:.4f}")
    print(f"    Accuracy:      {eval_results['accuracy']:.4f}")
    print(f"    Symmetry Error: {eval_results['symmetry_error']:.6f}")

    print("\n  Целевые показатели:")
    if eval_results['roc_auc'] > 0.68:
        print(f"    [OK] ROC-AUC > 0.68: DOSTIGNUT ({eval_results['roc_auc']:.4f})")
    else:
        print(f"    [FAIL] ROC-AUC > 0.68: NE DOSTIGNUT ({eval_results['roc_auc']:.4f})")

    if eval_results['symmetry_error'] < 0.05:
        print(f"    [OK] Simmetrichnost: {eval_results['symmetry_error']:.6f}")
    else:
        print(f"    [FAIL] Simmetrichnost: {eval_results['symmetry_error']:.6f}")

    print("\n" + "=" * 70)
    print("  SOHRANYYENNYYE FAYLY")
    print("=" * 70)
    print("""
  Данные:
    - features.csv              - признаки для обучения

  Модель:
    - model_lgbm.pkl            - обученная модель LightGBM

  Отчёты:
    - train_report.txt          - отчёт об обучении
    - evaluate_report.txt       - отчёт об оценке
    - calibration_curve.png     - калибровочная кривая
    - predictions_distribution.png - распределение предсказаний
    """)

    print("=" * 70)
    print("  PAYPLAYN ZAVYERSHYON")
    print("=" * 70)

    return eval_results


if __name__ == '__main__':
    main()
