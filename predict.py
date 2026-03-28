"""
Модуль для предсказания исхода конкретной карты между двумя командами.

Использование:
    from predict import predict_match, load_model

    model = load_model()
    result = predict_match(
        model,
        team1='Vitality',
        team2='Furia',
        map_name='Inferno',
        is_lan=1
    )
    print(f"Вероятность победы {team1}: {result['prob_team1']:.2%}")
"""
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import sqlite3

from utils import load_team_ratings, get_team_rating, step_weight, normalize_team_name


def load_data(db_path: str = 'cs2_data.db') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает данные из базы с нормализованными именами команд.
    """
    conn = sqlite3.connect(db_path)

    matches_df = pd.read_sql_query("""
        SELECT id, hltv_match_id, date, tournament, is_lan,
               team1_name, team2_name, team1_score, team2_score, winner
        FROM matches
        ORDER BY date
    """, conn)

    games_df = pd.read_sql_query("""
        SELECT id, match_id, map_name, team1_score, team2_score, y
        FROM games
    """, conn)

    conn.close()

    matches_df['date'] = pd.to_datetime(matches_df['date'])

    # Нормализуем имена команд
    matches_df['team1_name'] = matches_df['team1_name'].apply(normalize_team_name)
    matches_df['team2_name'] = matches_df['team2_name'].apply(normalize_team_name)

    return matches_df, games_df


def find_team_in_db(team_name: str, matches_df: pd.DataFrame) -> str:
    """
    Ищет команду в базе данных. Теперь все имена нормализованы (lowercase).
    Возвращает нормализованное имя команды.
    """
    normalized_input = normalize_team_name(team_name)

    all_teams = set(matches_df['team1_name'].tolist() + matches_df['team2_name'].tolist())

    # Ищем точное совпадение (после нормализации)
    if normalized_input in all_teams:
        return normalized_input

    # Если не найдено, возвращаем нормализованное имя (может быть 0 игр)
    return normalized_input


def load_model(model_path: str = 'model_lgbm.pkl'):
    """
    Загружает обученную модель LightGBM.
    """
    return joblib.load(model_path)


def compute_form_for_team(
    team_name: str,
    matches_df: pd.DataFrame,
    games_df: pd.DataFrame,
    target_date: datetime,
    window_maps: int = 10,
    ratings: Dict[str, float] = None
) -> Tuple[float, float, int]:
    """
    Вычисляет форму команды за последние window_maps карт.
    """
    team_matches = matches_df[
        ((matches_df['team1_name'] == team_name) | (matches_df['team2_name'] == team_name)) &
        (matches_df['date'] <= pd.Timestamp(target_date))
    ].copy()

    if len(team_matches) == 0:
        return np.nan, np.nan, 999

    team_matches = team_matches.sort_values('date', ascending=False)
    match_ids = team_matches['id'].tolist()
    team_games = games_df[games_df['match_id'].isin(match_ids)].copy()

    team_games = team_games.merge(
        team_matches[['id', 'date', 'team1_name', 'team2_name']],
        left_on='match_id',
        right_on='id',
        suffixes=('', '_match')
    )

    team_games['is_team1'] = team_games['team1_name'] == team_name
    team_games['y_from_team'] = np.where(
        team_games['is_team1'],
        team_games['y'],
        -team_games['y']
    )

    team_games = team_games.sort_values('date', ascending=False)
    recent_games = team_games.head(window_maps)

    if len(recent_games) == 0:
        return np.nan, np.nan, 999

    recent_games['days_ago'] = (target_date - recent_games['date']).dt.days
    recent_games['weight'] = recent_games['days_ago'].apply(step_weight)

    total_weight = recent_games['weight'].sum()
    if total_weight > 0:
        form = (recent_games['y_from_team'] * recent_games['weight']).sum() / total_weight
    else:
        form = np.nan

    if ratings is not None:
        def get_opponent_rating(row):
            if row['is_team1']:
                opponent = row['team2_name']
            else:
                opponent = row['team1_name']
            return get_team_rating(opponent, ratings, 0)

        recent_games['opponent_rating'] = recent_games.apply(get_opponent_rating, axis=1)

        if total_weight > 0:
            sos_form = (recent_games['opponent_rating'] * recent_games['weight']).sum() / total_weight
        else:
            sos_form = np.nan
    else:
        sos_form = np.nan

    last_match_date = team_matches['date'].max()
    days_since_last = (target_date - last_match_date).days

    return form, sos_form, days_since_last


def compute_map_stats_for_team(
    team_name: str,
    map_name: str,
    matches_df: pd.DataFrame,
    games_df: pd.DataFrame,
    target_date: datetime,
    window_days: int = 365
) -> Tuple[float, float, float, float, int]:
    """
    Вычисляет статистику команды на конкретной карте.
    """
    team_matches = matches_df[
        ((matches_df['team1_name'] == team_name) | (matches_df['team2_name'] == team_name)) &
        (matches_df['date'] <= pd.Timestamp(target_date))
    ].copy()

    if len(team_matches) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    match_ids = team_matches['id'].tolist()
    map_games = games_df[
        (games_df['match_id'].isin(match_ids)) &
        (games_df['map_name'] == map_name)
    ].copy()

    if len(map_games) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    map_games = map_games.merge(
        team_matches[['id', 'date', 'team1_name', 'team2_name']],
        left_on='match_id',
        right_on='id',
        suffixes=('', '_match')
    )

    map_games['is_team1'] = map_games['team1_name'] == team_name
    map_games['y_from_team'] = np.where(
        map_games['is_team1'],
        map_games['y'],
        -map_games['y']
    )

    map_games['t_side_perf'] = np.where(
        map_games['is_team1'],
        map_games['team1_score'],
        map_games['team2_score']
    )
    map_games['ct_side_perf'] = np.where(
        map_games['is_team1'],
        map_games['team2_score'],
        map_games['team1_score']
    )

    map_games['days_ago'] = (target_date - map_games['date']).dt.days
    map_games_12m = map_games[map_games['days_ago'] <= window_days]

    if len(map_games_12m) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    map_games_12m = map_games_12m.copy()
    map_games_12m['weight'] = map_games_12m['days_ago'].apply(step_weight)
    total_weight_12m = map_games_12m['weight'].sum()

    if total_weight_12m > 0:
        map_strength = (map_games_12m['y_from_team'] * map_games_12m['weight']).sum() / total_weight_12m
        map_t_winrate = map_games_12m['t_side_perf'].mean() / 24.0
        map_ct_winrate = map_games_12m['ct_side_perf'].mean() / 24.0
    else:
        map_strength = np.nan
        map_t_winrate = np.nan
        map_ct_winrate = np.nan

    map_games_3m = map_games[map_games['days_ago'] <= 90].copy()

    if len(map_games_3m) > 0:
        map_games_3m['weight'] = map_games_3m['days_ago'].apply(step_weight)
        total_weight_3m = map_games_3m['weight'].sum()
        if total_weight_3m > 0:
            map_recent_form = (map_games_3m['y_from_team'] * map_games_3m['weight']).sum() / total_weight_3m
        else:
            map_recent_form = np.nan
    else:
        map_recent_form = np.nan

    map_count = len(map_games_12m)

    return map_strength, map_recent_form, map_t_winrate, map_ct_winrate, map_count


def compute_recent_matches_count(
    team_name: str,
    matches_df: pd.DataFrame,
    target_date: datetime,
    window_days: int = 14
) -> int:
    """
    Количество матчей команды за последние window_days дней.
    """
    recent = matches_df[
        ((matches_df['team1_name'] == team_name) | (matches_df['team2_name'] == team_name)) &
        (matches_df['date'] <= pd.Timestamp(target_date)) &
        (matches_df['date'] >= pd.Timestamp(target_date) - pd.Timedelta(days=window_days))
    ]
    return len(recent)


def find_team_in_db(team_name: str, matches_df: pd.DataFrame) -> str:
    """
    Ищет команду в базе данных с учётом регистра.
    Возвращает найденное имя или оригинальное.
    """
    all_teams = set(matches_df['team1_name'].tolist() + matches_df['team2_name'].tolist())

    # Пробуем найти точное совпадение
    if team_name in all_teams:
        return team_name

    # Пробуем варианты регистра
    variants = [
        team_name.upper(),
        team_name.lower(),
        team_name.capitalize(),
    ]
    for variant in variants:
        if variant in all_teams:
            return variant

    # Возвращаем оригинал (не найдено)
    return team_name


def predict_match(
    model,
    team1: str,
    team2: str,
    map_name: str,
    is_lan: int = 0,
    reference_date: datetime = None,
    db_path: str = 'cs2_data.db',
    ratings_path: str = 'top_teams.txt'
) -> Dict:
    """
    Предсказывает исход матча между двумя командами на конкретной карте.
    Все имена команд автоматически нормализуются (lowercase + маппинг).

    Args:
        model: обученная модель LightGBM
        team1: название первой команды (например, 'Vitality', 'Furia', 'NAVI')
        team2: название второй команды
        map_name: название карты
        is_lan: 1 для LAN турнира, 0 для онлайн
        reference_date: дата матча (по умолчанию сегодня)
        db_path: путь к базе данных
        ratings_path: путь к файлу с рейтингами команд

    Returns:
        Dict с результатами:
        - prob_team1: вероятность победы team1
        - prob_team2: вероятность победы team2
        - prediction: предсказанный исход (1 или 2)
        - features: использованные признаки
    """
    if reference_date is None:
        reference_date = datetime.now()

    # Загружаем данные (имена уже нормализованы в load_data)
    ratings = load_team_ratings(ratings_path)
    matches_df, games_df = load_data(db_path)

    # Нормализуем входные имена команд
    team1_norm = normalize_team_name(team1)
    team2_norm = normalize_team_name(team2)

    # Проверяем, есть ли команды в базе
    all_teams = set(matches_df['team1_name'].tolist() + matches_df['team2_name'].tolist())

    if team1_norm not in all_teams:
        print(f"  [WARN] Команда '{team1}' ({team1_norm}) не найдена в базе данных")
    if team2_norm not in all_teams:
        print(f"  [WARN] Команда '{team2}' ({team2_norm}) не найдена в базе данных")

    # Вычисляем признаки для team1
    rating1 = get_team_rating(team1_norm, ratings, 0)
    rating2 = get_team_rating(team2_norm, ratings, 0)
    delta_rating = rating1 - rating2

    form1, sos1, days1 = compute_form_for_team(
        team1_norm, matches_df, games_df, reference_date, 10, ratings
    )
    form2, sos2, days2 = compute_form_for_team(
        team2_norm, matches_df, games_df, reference_date, 10, ratings
    )
    form_diff_10 = form1 - form2
    sos_form_diff_10 = sos1 - sos2
    days_since_last_match_diff = min(days1 - days2, 30)

    matches1 = compute_recent_matches_count(team1_norm, matches_df, reference_date, 14)
    matches2 = compute_recent_matches_count(team2_norm, matches_df, reference_date, 14)
    recent_matches_count_diff = matches1 - matches2

    ms1, mrf1, mt1, mct1, mc1 = compute_map_stats_for_team(
        team1_norm, map_name, matches_df, games_df, reference_date, 365
    )
    ms2, mrf2, mt2, mct2, mc2 = compute_map_stats_for_team(
        team2_norm, map_name, matches_df, games_df, reference_date, 365
    )
    map_strength_diff = ms1 - ms2
    map_recent_form_diff = mrf1 - mrf2
    map_t_side_winrate_diff = mt1 - mt2
    map_ct_side_winrate_diff = mct1 - mct2
    map_sample_diff = np.log(mc1 + 1) - np.log(mc2 + 1)

    # Взаимодействия
    rating_x_map_strength = delta_rating * map_strength_diff
    form_x_map_form = form_diff_10 * map_recent_form_diff
    rating_x_sos = delta_rating * sos_form_diff_10

    # Заполняем NaN нулями
    def safe_fill_nan(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0
        return float(val)

    delta_rating = safe_fill_nan(delta_rating)
    form_diff_10 = safe_fill_nan(form_diff_10)
    sos_form_diff_10 = safe_fill_nan(sos_form_diff_10)
    days_since_last_match_diff = safe_fill_nan(days_since_last_match_diff)
    recent_matches_count_diff = float(recent_matches_count_diff)
    map_strength_diff = safe_fill_nan(map_strength_diff)
    map_recent_form_diff = safe_fill_nan(map_recent_form_diff)
    map_t_side_winrate_diff = safe_fill_nan(map_t_side_winrate_diff)
    map_ct_side_winrate_diff = safe_fill_nan(map_ct_side_winrate_diff)
    map_sample_diff = safe_fill_nan(map_sample_diff)
    rating_x_map_strength = safe_fill_nan(rating_x_map_strength)
    form_x_map_form = safe_fill_nan(form_x_map_form)
    rating_x_sos = safe_fill_nan(rating_x_sos)

    # Создаём DataFrame с признаками
    feature_cols = [
        'delta_rating', 'form_diff_10', 'sos_form_diff_10',
        'days_since_last_match_diff', 'recent_matches_count_diff',
        'map_strength_diff', 'map_recent_form_diff',
        'map_t_side_winrate_diff', 'map_ct_side_winrate_diff', 'map_sample_diff',
        'rating_x_map_strength', 'form_x_map_form', 'rating_x_sos'
    ]

    X = pd.DataFrame({
        'delta_rating': [delta_rating],
        'form_diff_10': [form_diff_10],
        'sos_form_diff_10': [sos_form_diff_10],
        'days_since_last_match_diff': [days_since_last_match_diff],
        'recent_matches_count_diff': [recent_matches_count_diff],
        'map_strength_diff': [map_strength_diff],
        'map_recent_form_diff': [map_recent_form_diff],
        'map_t_side_winrate_diff': [map_t_side_winrate_diff],
        'map_ct_side_winrate_diff': [map_ct_side_winrate_diff],
        'map_sample_diff': [map_sample_diff],
        'rating_x_map_strength': [rating_x_map_strength],
        'form_x_map_form': [form_x_map_form],
        'rating_x_sos': [rating_x_sos]
    })

    # Предсказание
    prob_team1 = model.predict_proba(X)[0, 1]
    prob_team2 = 1 - prob_team1
    prediction = 1 if prob_team1 >= 0.5 else 2

    return {
        'prob_team1': prob_team1,
        'prob_team2': prob_team2,
        'prediction': prediction,
        'features': X.to_dict('records')[0]
    }


def predict_match_swapped(
    model,
    team1: str,
    team2: str,
    map_name: str,
    is_lan: int = 0,
    reference_date: datetime = None,
    db_path: str = 'cs2_data.db',
    ratings_path: str = 'top_teams.txt'
) -> Tuple[Dict, Dict]:
    """
    Предсказывает исход матча и проверяет симметричность.

    Returns:
        Tuple с результатами для (team1, team2) и (team2, team1)
    """
    result_ab = predict_match(
        model, team1, team2, map_name, is_lan, reference_date, db_path, ratings_path
    )
    result_ba = predict_match(
        model, team2, team1, map_name, is_lan, reference_date, db_path, ratings_path
    )

    return result_ab, result_ba


if __name__ == '__main__':
    # Пример использования
    print("Загрузка модели...")
    model = load_model()

    team1_name = "PARIVISION"
    team2_name = "Falcons"
    map_name = "Anubis"

    print(f"\nПредсказание для матча: {team1_name} vs {team2_name} на карте {map_name}")
    result = predict_match(
        model,
        team1=team1_name,
        team2=team2_name,
        map_name=map_name,
        is_lan=1
    )

    print(f"  Вероятность победы {team1_name} (team1): {result['prob_team1']:.2%}")
    print(f"  Вероятность победы {team2_name} (team2):    {result['prob_team2']:.2%}")
    print(f"  Предсказание: Победа {team1_name if result['prediction'] == 1 else team2_name}")
