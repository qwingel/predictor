"""
Вспомогательные функции для системы предсказания карт CS2.
"""
import re
from datetime import datetime, timedelta
from typing import Dict, Optional
import numpy as np


# Маппинг известных вариаций команд (оригинал -> нормализованный)
TEAM_MAPPING = {
    # Natus Vincere
    'navi': 'natus vincere',
    'navi junior': 'natus vincere junior',
    'ex-navi junior': 'ex-natus vincere junior',

    # FaZe
    'faze': 'faze',

    # Другие возможные вариации
    'virtus.pro': 'virtus pro',
    'team spirit': 'spirit',
    'furía': 'furia',
}


def normalize_team_name(team_name: str) -> str:
    """
    Нормализует название команды:
    1. Приводит к lowercase
    2. Применяет маппинг известных вариаций
    """
    lower = team_name.lower().strip()
    return TEAM_MAPPING.get(lower, lower)


def load_team_ratings(filepath: str = "data/top_teams.txt") -> Dict[str, float]:
    """
    Загружает рейтинги команд из top_teams.txt с нормализованными именами.

    Формат файла:
    1: "Vitality", 1000
    2: "Furia", 561

    Возвращает словарь: {normalized_team_name: points}
    Для отсутствующих команд рейтинг будет 0.
    """
    ratings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Парсим: 1: "Vitality", 1000
            match = re.match(r'\d+:\s*"([^"]+)",\s*(\d+)', line)
            if match:
                team_name = match.group(1)
                points = float(match.group(2))
                # Нормализуем имя команды
                normalized_name = normalize_team_name(team_name)
                ratings[normalized_name] = points
    return ratings


def get_team_rating(team_name: str, ratings: Dict[str, float], default: float = 0.0) -> float:
    """
    Возвращает рейтинг команды. Если команда не найдена - default.
    """
    return ratings.get(team_name, default)


def step_weight(days_ago: float) -> float:
    """
    Возвращает вес матча по ступенчатой схеме:

    0-90 дней: 1.0
    90-180 дней: 0.7
    180-365 дней: 0.3
    старше 365 дней: 0.0
    """
    if days_ago < 0:
        return 0.0
    elif days_ago <= 7:
        return 1.5
    elif days_ago <= 30:
        return 1.2
    elif days_ago <= 90:
        return 1.0
    elif days_ago <= 180:
        return 0.7
    elif days_ago <= 365:
        return 0.3
    else:
        return 0.0


def train_step_weight(days_ago: float) -> float:
    """
    Веса для обучения модели (по ТЗ):

    последние 3 месяца – вес 1.0
    3–6 месяцев – вес 0.7
    6–12 месяцев – вес 0.3
    старше 12 месяцев – не использовать (вес 0)
    """
    return step_weight(days_ago)


def rolling_weighted_mean(
    data: 'pd.DataFrame',
    date_col: str,
    value_col: str,
    weight_col: Optional[str],
    window_days: int,
    target_date: datetime,
    step_weights: bool = True
) -> float:
    """
    Вычисляет взвешенное скользящее среднее для данных за последние window_days дней.
    """
    import pandas as pd

    # Фильтруем данные только до target_date
    mask = pd.to_datetime(data[date_col]) <= pd.Timestamp(target_date)
    filtered = data[mask].copy()

    if len(filtered) == 0:
        return np.nan

    # Вычисляем days_ago для каждой записи
    filtered['days_ago'] = (target_date - pd.to_datetime(filtered[date_col])).dt.days

    # Фильтруем по окну
    filtered = filtered[filtered['days_ago'] <= window_days]

    if len(filtered) == 0:
        return np.nan

    # Вычисляем веса
    if weight_col is None:
        if step_weights:
            filtered['w'] = filtered['days_ago'].apply(step_weight)
        else:
            # Линейные веса: более свежие = больший вес
            filtered['w'] = 1.0 / (filtered['days_ago'] + 1)
    else:
        filtered['w'] = filtered[weight_col]

    # Взвешенное среднее
    values = filtered[value_col].values
    weights = filtered['w'].values

    if np.sum(weights) == 0:
        return np.nan

    return np.sum(values * weights) / np.sum(weights)


def parse_date(date_str: str) -> datetime:
    """
    Парсит дату из строки формата YYYY-MM-DD.
    """
    return datetime.strptime(date_str, '%Y-%m-%d')


def days_between(date1: datetime, date2: datetime) -> int:
    """
    Возвращает количество дней между двумя датами.
    """
    return abs((date2 - date1).days)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Сигмоидная функция для преобразования вероятности в логиты.
    """
    return 1 / (1 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    """
    Логит функция (обратная сигмоиде).
    """
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))
