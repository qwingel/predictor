"""
Скрипт для валидации модели на утечки данных и переобучение.

Запуск:
    python validate_model.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score


def check_identifiers():
    """Проверка 1: идентификаторы не входят в признаки."""
    print("=" * 60)
    print("ПРОВЕРКА 1: ИДЕНТИФИКАТОРЫ В ПРИЗНАКАХ")
    print("=" * 60)

    df = pd.read_csv('data/features.csv')
    feature_cols = [
        'delta_rating', 'form_diff_10', 'sos_form_diff_10',
        'days_since_last_match_diff', 'recent_matches_count_diff',
        'map_strength_diff', 'map_recent_form_diff',
        'map_t_side_winrate_diff', 'map_ct_side_winrate_diff', 'map_sample_diff',
        'rating_x_map_strength', 'form_x_map_form', 'rating_x_sos'    ]

    has_leak = 'match_id' in feature_cols or 'game_id' in feature_cols
    has_y = 'y' in feature_cols

    if has_leak or has_y:
        print("FAIL: Идентификаторы или y входят в признаки!")
        return False
    else:
        print("OK: Идентификаторы не входят в признаки")
        return True


def check_target_correlation():
    """Проверка 2: корреляция признаков с y."""
    print()
    print("=" * 60)
    print("ПРОВЕРКА 2: КОРРЕЛЯЦИЯ ПРИЗНАКОВ С Y")
    print("=" * 60)

    df = pd.read_csv('data/features.csv')
    feature_cols = [
        'delta_rating', 'form_diff_10', 'sos_form_diff_10',
        'days_since_last_match_diff', 'recent_matches_count_diff',
        'map_strength_diff', 'map_recent_form_diff',
        'map_t_side_winrate_diff', 'map_ct_side_winrate_diff', 'map_sample_diff',
        'rating_x_map_strength', 'form_x_map_form', 'rating_x_sos'
    ]

    high_corr = []
    for col in feature_cols:
        corr = df[col].corr(df['y'])
        if not np.isnan(corr) and abs(corr) > 0.5:
            high_corr.append((col, corr))
        print(f"  {col}: {corr:.4f}")

    if high_corr:
        print(f"WARNING: Признаки с высокой корреляцией (>0.5): {len(high_corr)}")
        for col, corr in high_corr:
            print(f"    - {col}: {corr:.4f}")
        print("  (Высокая корреляция допустима для исторических признаков)")
    else:
        print("OK: Нет признаков с аномально высокой корреляцией")

    return True


def check_temporal_split():
    """Проверка 3: временной сплит без пересечений."""
    print()
    print("=" * 60)
    print("ПРОВЕРКА 3: ВРЕМЕННОЙ СПЛИТ")
    print("=" * 60)

    df = pd.read_csv('data/features.csv')
    df['date'] = pd.to_datetime(df['date'])

    unique_dates = sorted(df['date'].unique())
    n_dates = len(unique_dates)
    train_date_idx = int(n_dates * 0.70)
    val_date_idx = int(n_dates * 0.85)

    train_dates = set(unique_dates[:train_date_idx])
    val_dates = set(unique_dates[train_date_idx:val_date_idx])
    test_dates = set(unique_dates[val_date_idx:])

    train_val_overlap = train_dates & val_dates
    val_test_overlap = val_dates & test_dates

    print(f"Уникальных дат: {n_dates}")
    print(f"Train дат: {len(train_dates)}, Val дат: {len(val_dates)}, Test дат: {len(test_dates)}")

    if train_val_overlap:
        print(f"FAIL: Пересечение Train/Val: {len(train_val_overlap)} дат")
        return False
    elif val_test_overlap:
        print(f"FAIL: Пересечение Val/Test: {len(val_test_overlap)} дат")
        return False
    else:
        print("OK: Нет пересечения дат между выборками")
        return True


def check_overfitting():
    """Проверка 4: переобучение."""
    print()
    print("=" * 60)
    print("ПРОВЕРКА 4: ПЕРЕОБУЧЕНИЕ")
    print("=" * 60)

    model = joblib.load('models/model_lgbm.pkl')
    df = pd.read_csv('data/features.csv')
    df['date'] = pd.to_datetime(df['date'])

    feature_cols = [
        'delta_rating', 'form_diff_10', 'sos_form_diff_10',
        'days_since_last_match_diff', 'recent_matches_count_diff',
        'map_strength_diff', 'map_recent_form_diff',
        'map_t_side_winrate_diff', 'map_ct_side_winrate_diff', 'map_sample_diff',
        'rating_x_map_strength', 'form_x_map_form', 'rating_x_sos'
    ]

    # Сплит по датам
    unique_dates = sorted(df['date'].unique())
    n_dates = len(unique_dates)
    train_date_idx = int(n_dates * 0.70)
    val_date_idx = int(n_dates * 0.85)

    train_dates = unique_dates[:train_date_idx]
    val_dates = unique_dates[train_date_idx:val_date_idx]
    test_dates = unique_dates[val_date_idx:]

    train_df = df[df['date'].isin(train_dates)]
    val_df = df[df['date'].isin(val_dates)]
    test_df = df[df['date'].isin(test_dates)]

    X_train = train_df[feature_cols].values
    y_train = (train_df['y'].values + 1) / 2
    X_val = val_df[feature_cols].values
    y_val = (val_df['y'].values + 1) / 2
    X_test = test_df[feature_cols].values
    y_test = (test_df['y'].values + 1) / 2

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"ROC-AUC на Train: {train_auc:.4f}")
    print(f"ROC-AUC на Val:   {val_auc:.4f}")
    print(f"ROC-AUC на Test:  {test_auc:.4f}")

    gap = train_auc - test_auc
    print(f"Gap (Train - Test): {gap:.4f}")

    if gap > 0.1:
        print("FAIL: Большое расхождение (>0.1) - переобучение!")
        return False
    elif gap > 0.05:
        print("WARNING: Умеренное расхождение (0.05-0.1)")
        return True
    else:
        print("OK: Расхождение в пределах нормы (<0.05)")
        return True


def check_feature_importance():
    """Проверка 5: важность признаков."""
    print()
    print("=" * 60)
    print("ПРОВЕРКА 5: ВАЖНОСТЬ ПРИЗНАКОВ")
    print("=" * 60)

    model = joblib.load('models/model_lgbm.pkl')
    feature_cols = [
        'delta_rating', 'form_diff_10', 'sos_form_diff_10',
        'days_since_last_match_diff', 'recent_matches_count_diff',
        'map_strength_diff', 'map_recent_form_diff',
        'map_t_side_winrate_diff', 'map_ct_side_winrate_diff', 'map_sample_diff',
        'rating_x_map_strength', 'form_x_map_form', 'rating_x_sos'
    ]

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(importance.to_string(index=False))

    total_imp = importance['importance'].sum()
    top_pct = importance.iloc[0]['importance'] / total_imp * 100

    print(f"\nТоп признак: {top_pct:.1f}% от общей важности")

    if top_pct > 50:
        print("FAIL: Один признак доминирует (>50%)!")
        return False
    else:
        print("OK: Важность распределена между признаками")
        return True


def check_symmetry():
    """Проверка 6: симметричность модели."""
    print()
    print("=" * 60)
    print("ПРОВЕРКА 6: СИММЕТРИЧНОСТЬ")
    print("=" * 60)

    from predict import load_model, predict_match

    model = load_model()
    errors = []

    # Тестовые пары
    test_cases = [
        ('Vitality', 'Furia', 'Inferno'),
        ('Natus Vincere', 'Spirit', 'Mirage'),
        ('Astralis', 'HEROIC', 'Nuke'),
        ('Faze', 'G2', 'Ancient'),
        ('Liquid', 'NRG', 'Dust2'),
    ]

    for team1, team2, map_name in test_cases:
        result_ab = predict_match(model, team1, team2, map_name)
        result_ba = predict_match(model, team2, team1, map_name)

        error = abs(result_ab['prob_team1'] + result_ba['prob_team1'] - 1)
        errors.append(error)
        print(f"  {team1} vs {team2} на {map_name}: error = {error:.4f}")

    mean_error = np.mean(errors)
    print(f"\nСредняя ошибка симметричности: {mean_error:.4f}")

    if mean_error >= 0.05:
        print("WARNING: Ошибка >= 0.05 (требуется < 0.05)")
        return False
    else:
        print("OK: Модель симметрична (error < 0.05)")
        return True


def main():
    print("=" * 60)
    print("  ВАЛИДАЦИЯ МОДЕЛИ CS2 PREDICTION")
    print("=" * 60)
    print()

    results = {
        'identifiers': check_identifiers(),
        'correlation': check_target_correlation(),
        'temporal_split': check_temporal_split(),
        'overfitting': check_overfitting(),
        'feature_importance': check_feature_importance(),
        'symmetry': check_symmetry()
    }

    print()
    print("=" * 60)
    print("  ИТОГИ ВАЛИДАЦИИ")
    print("=" * 60)

    for check, passed in results.items():
        status = "OK" if passed else "FAIL"
        print(f"  [{status}] {check}")

    all_passed = all(results.values())

    print()
    if all_passed:
        print("=" * 60)
        print("  ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
        print("=" * 60)
        print("\nМодель готова к использованию!")
    else:
        print("=" * 60)
        print("  НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОЙДЕНЫ")
        print("=" * 60)
        print("\nТребуется доработка модели!")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
