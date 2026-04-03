"""
Модуль для обучения модели предсказания исхода карт CS2.

Алгоритм: LightGBM с boosting_type='goss'
Симметричная модель: все признаки - разница между командами.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

from utils import train_step_weight


def load_features(features_path: str = '../data/features.csv') -> pd.DataFrame:
    """
    Загружает признаки из CSV файла.
    """
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def add_match_weights(df: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
    """
    Добавляет веса матчей для обучения по ступенчатой схеме.

    последние 3 месяца – вес 1.0
    3–6 месяцев – вес 0.7
    6–12 месяцев – вес 0.3
    старше 12 месяцев – вес 0.0
    """
    if reference_date is None:
        reference_date = df['date'].max()

    df = df.copy()
    df['days_ago'] = (reference_date - df['date']).dt.days
    df['sample_weight'] = df['days_ago'].apply(train_step_weight)

    return df


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Делает временной сплит по уникальным датам.
    Важно: все матчи одной даты попадают в одну выборку.
    """
    df_sorted = df.sort_values('date').reset_index(drop=True)

    # Получаем уникальные даты
    unique_dates = sorted(df_sorted['date'].unique())
    n_dates = len(unique_dates)

    # Сплит по датам (не по записям)
    train_date_idx = int(n_dates * train_ratio)
    val_date_idx = int(n_dates * (train_ratio + val_ratio))

    train_dates = unique_dates[:train_date_idx]
    val_dates = unique_dates[train_date_idx:val_date_idx]
    test_dates = unique_dates[val_date_idx:]

    # Разделяем данные по датам
    train_df = df_sorted[df_sorted['date'].isin(train_dates)].copy()
    val_df = df_sorted[df_sorted['date'].isin(val_dates)].copy()
    test_df = df_sorted[df_sorted['date'].isin(test_dates)].copy()

    print(f"Временной сплит (по уникальным датам):")
    print(f"  Train: {len(train_df)} ({len(train_dates)} дат), даты: {train_df['date'].min()} - {train_df['date'].max()}")
    print(f"  Val:   {len(val_df)} ({len(val_dates)} дат), даты: {val_df['date'].min()} - {val_df['date'].max()}")
    print(f"  Test:  {len(test_df)} ({len(test_dates)} дат), даты: {test_df['date'].min()} - {test_df['date'].max()}")

    # Проверка на пересечение дат
    assert len(set(train_dates) & set(val_dates)) == 0, "Train/Val пересечение!"
    assert len(set(val_dates) & set(test_dates)) == 0, "Val/Test пересечение!"
    print("  Проверка: OK (нет пересечения дат)")

    return train_df, val_df, test_df


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


def prepare_data(
    df: pd.DataFrame,
    feature_cols: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Подготавливает данные для обучения.

    Returns:
        X: признаки
        y: целевая переменная (преобразована из -1/1 в 0/1 для binary classification)
        weights: веса样本
    """
    X = df[feature_cols].values

    # Преобразуем y из -1/1 в 0/1 для LightGBM binary classification
    y_binary = (df['y'].values + 1) / 2  # -1 -> 0, 1 -> 1

    weights = df['sample_weight'].values

    return X, y_binary, weights


def train_model(
    train_df: pd.DataFrame,
    params: Dict = None
) -> lgb.Booster:
    """
    Обучает модель LightGBM.
    """
    if params is None:
        params = {
            'boosting_type': 'goss',
            'n_estimators': 900,
            'learning_rate': 0.01,
            'num_leaves': 18,
            'max_depth': 4,
            'reg_lambda': 15,
            'reg_alpha': 1.5,
            'min_child_samples': 40,
            'colsample_bytree': 0.78,
            'random_state': 42,
            'objective': 'binary',
            'metric': 'auc',
            'verbose': -1
        }

    feature_cols = get_feature_columns()

    # Подготавливаем данные
    X_train, y_train, w_train = prepare_data(train_df, feature_cols)

    print("\nОбучение модели LightGBM с параметрами:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Обучаем модель
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model: lgb.LGBMClassifier,
    test_df: pd.DataFrame,
    prefix: str = "Test"
) -> Dict[str, float]:
    """
    Вычисляет метрики на тестовой выборке.
    """
    feature_cols = get_feature_columns()
    X_test, y_test, _ = prepare_data(test_df, feature_cols)

    # Предсказания (вероятности класса 1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Предсказания классов
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Метрики
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{prefix} метрики:")
    print(f"  ROC-AUC:  {roc_auc:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

    return {
        'roc_auc': roc_auc,
        'brier': brier,
        'accuracy': accuracy
    }


def get_feature_importance(model: lgb.LGBMClassifier) -> pd.DataFrame:
    """
    Возвращает важность признаков.
    """
    feature_cols = get_feature_columns()
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    return importance


def train_and_save(
    features_path: str = '../data/features.csv',
    model_path: str = '../models/model_lgbm_final.pkl',
    report_path: str = '../data/final_train_report.txt'
) -> Tuple[lgb.LGBMClassifier, Dict]:
    """
    Полный пайплайн обучения и сохранения модели.

    Returns:
        model: обученная модель
        metrics: метрики на тесте
    """
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ PREDICTION CS2")
    print("=" * 60)

    # Загружаем данные
    print("\nЗагрузка данных...")
    df = load_features(features_path)
    print(f"Загружено {len(df)} записей")

    # Добавляем веса
    print("\nДобавление весов матчей...")
    df = add_match_weights(df)
    print(f"Записей с ненулевым весом: {(df['sample_weight'] > 0).sum()}")

    # Фильтруем записи с нулевым весом
    df = df[df['sample_weight'] > 0].copy()

    # Временной сплит
    print("\nВременной сплит данных...")

    train_df = df.copy() # обучения ПО ВСЕЙ выборке

    print(f"  Train: {len(train_df)} записей")

    # Обучение модели
    print("\nОбучение модели...")
    model = train_model(train_df)

    # Сохраняем модель
    joblib.dump(model, model_path)
    print(f"\nМодель сохранена в {model_path}")

    # Сохраняем отчёт
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ОТЧЁТ ОБ ОБУЧЕНИИ МОДЕЛИ\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("ПАРАМЕТРЫ МОДЕЛИ:\n")
        f.write("-" * 40 + "\n")
        params = model.get_params()
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")

    print(f"Отчёт сохранён в {report_path}")

    return model

if __name__ == '__main__':
    model = train_and_save()
