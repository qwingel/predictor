"""
Модуль для построения признаков (features) для предсказания исхода карт CS2.

Все признаки симметричные: delta = team1_value - team2_value.
При перестановке команд местами все дельты инвертируются.
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from utils import load_team_ratings, get_team_rating, step_weight, parse_date, normalize_team_name


def load_data(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает данные из SQLite базы с нормализованными именами команд.

    Returns:
        matches_df: DataFrame с матчами (team1_name, team2_name нормализованы)
        games_df: DataFrame с картами
    """
    conn = sqlite3.connect(db_path)

    matches_df = pd.read_sql_query("""
        SELECT id, hltv_match_id, date, tournament, is_lan,
               team1_name, team2_name, team1_score, team2_score, winner
        FROM matches
        ORDER BY date
    """, conn)

    games_df = pd.read_sql_query("""
        SELECT id as game_id, match_id, map_name, team1_score, team2_score, y
        FROM games
    """, conn)

    conn.close()

    # Преобразуем даты
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    matches_df['is_lan'] = matches_df['is_lan'].astype(int)

    # Нормализуем имена команд (lowercase + маппинг)
    matches_df['team1_name'] = matches_df['team1_name'].apply(normalize_team_name)
    matches_df['team2_name'] = matches_df['team2_name'].apply(normalize_team_name)

    return matches_df, games_df


def compute_form_for_team(
    team_name: str,
    matches_df: pd.DataFrame,
    games_df: pd.DataFrame,
    target_date: datetime,
    window_maps: int = 10,
    ratings: Dict[str, float] = None,
    exclude_match_id: int = None   # <-- добавить
) -> Tuple[float, float, int]:
    """
    Вычисляет форму команды за последние window_maps карт.

    Args:
        team_name: имя команды
        matches_df: DataFrame с матчами
        games_df: DataFrame с картами
        target_date: дата, относительно которой считаем прошлое
        window_maps: количество последних карт для учёта
        ratings: словарь рейтингов команд (для SOS)

    Returns:
        form: взвешенный winrate (среднее взвешенное y)
        sos_form: сила формы с учётом соперников (средний рейтинг * вес)
        days_since_last: дней с последнего матча
    """
    # Находим все матчи команды до target_date
    team_matches = matches_df[
        ((matches_df['team1_name'] == team_name) | (matches_df['team2_name'] == team_name)) &
        (matches_df['date'] <= pd.Timestamp(target_date))
    ].copy()

    if exclude_match_id is not None:
        team_matches = team_matches[team_matches['id'] != exclude_match_id]

    if len(team_matches) == 0:
        return np.nan, np.nan, 999

    # Сортируем по дате
    team_matches = team_matches.sort_values('date', ascending=False)

    # Находим все игры команды
    match_ids = team_matches['id'].tolist()
    team_games = games_df[games_df['match_id'].isin(match_ids)].copy()

    # Добавляем информацию о матчах
    team_games = team_games.merge(
        team_matches[['id', 'date', 'team1_name', 'team2_name']],
        left_on='match_id',
        right_on='id',
        suffixes=('', '_match')
    )

    # Определяем, была ли команда team1 и какой результат с её перспективы
    team_games['is_team1'] = team_games['team1_name'] == team_name
    team_games['y_from_team'] = np.where(
        team_games['is_team1'],
        team_games['y'],
        -team_games['y']  # Если команда была team2, инвертируем
    )

    # Сортируем по дате матча (от новых к старым)
    team_games = team_games.sort_values('date', ascending=False)

    # Берём последние window_maps карт
    recent_games = team_games.head(window_maps)

    if len(recent_games) == 0:
        return np.nan, np.nan, 999

    # Вычисляем days_ago для каждой карты
    recent_games['days_ago'] = (target_date - recent_games['date']).dt.days

    # Вычисляем веса (ступенчатые)
    recent_games['weight'] = recent_games['days_ago'].apply(step_weight)

    # Взвешенная форма (среднее y с весами)
    total_weight = recent_games['weight'].sum()
    if total_weight > 0:
        form = (recent_games['y_from_team'] * recent_games['weight']).sum() / total_weight
    else:
        form = np.nan

    # SOS: средний рейтинг соперников с весами
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

    # Дней с последнего матча
    last_match_date = team_matches['date'].max()
    days_since_last = (target_date - last_match_date).days

    return form, sos_form, days_since_last


def compute_map_stats_for_team(
    team_name: str,
    map_name: str,
    matches_df: pd.DataFrame,
    games_df: pd.DataFrame,
    target_date: datetime,
    window_days: int = 365,
    exclude_match_id: int = None   # <-- добавить
) -> Tuple[float, float, float, float, int]:
    """
    Вычисляет статистику команды на конкретной карте.

    Args:
        team_name: имя команды
        map_name: название карты
        matches_df: DataFrame с матчами
        games_df: DataFrame с картами
        target_date: дата, относительно которой считаем прошлое
        window_days: окно в днях

    Returns:
        map_strength: взвешенный winrate на карте за 12 месяцев
        map_recent_form: взвешенный winrate на карте за 3 месяца
        map_t_winrate: взвешенный T-side winrate
        map_ct_winrate: взвешенный CT-side winrate
        map_count: количество сыгранных карт
    """
    # Находим матчи команды до target_date
    team_matches = matches_df[
        ((matches_df['team1_name'] == team_name) | (matches_df['team2_name'] == team_name)) &
        (matches_df['date'] <= pd.Timestamp(target_date))
    ].copy()

    if exclude_match_id is not None:
        team_matches = team_matches[team_matches['id'] != exclude_match_id]

    if len(team_matches) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    # Находим игры на этой карте
    match_ids = team_matches['id'].tolist()
    map_games = games_df[
        (games_df['match_id'].isin(match_ids)) &
        (games_df['map_name'] == map_name)
    ].copy()

    if len(map_games) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    # Добавляем информацию о матчах
    map_games = map_games.merge(
        team_matches[['id', 'date', 'team1_name', 'team2_name']],
        left_on='match_id',
        right_on='id',
        suffixes=('', '_match')
    )

    # Определяем позицию команды и результат
    map_games['is_team1'] = map_games['team1_name'] == team_name
    map_games['y_from_team'] = np.where(
        map_games['is_team1'],
        map_games['y'],
        -map_games['y']
    )

    # T-side: team1 играет за T (условно, для простоты)
    # В реальности нужно знать, кто был за какой стороной
    # Для упрощения считаем, что team1_score - это score T-side
    # На самом деле в CS2 стороны меняются, но для оценки можно использовать
    # разницу раундов как прокси для T-side эффективности
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

    # Вычисляем days_ago
    map_games['days_ago'] = (target_date - map_games['date']).dt.days

    # Фильтруем по окну 12 месяцев
    map_games_12m = map_games[map_games['days_ago'] <= window_days]

    if len(map_games_12m) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    # Веса для 12 месяцев
    map_games_12m = map_games_12m.copy()
    map_games_12m['weight'] = map_games_12m['days_ago'].apply(step_weight)

    total_weight_12m = map_games_12m['weight'].sum()

    if total_weight_12m > 0:
        map_strength = (map_games_12m['y_from_team'] * map_games_12m['weight']).sum() / total_weight_12m

        # T-side и CT-side winrate (используем перформанс как прокси)
        # Нормализуем: средний score ~13, делим на 26 (максимум раундов в MR12)
        map_t_winrate = map_games_12m['t_side_perf'].mean() / 24.0
        map_ct_winrate = map_games_12m['ct_side_perf'].mean() / 24.0
    else:
        map_strength = np.nan
        map_t_winrate = np.nan
        map_ct_winrate = np.nan

    # Recent form на карте (3 месяца = 90 дней)
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
    window_days: int = 14,
    exclude_match_id: int = None
) -> int:
    """
    Количество матчей команды за последние window_days дней.
    """
    recent = matches_df[
        ((matches_df['team1_name'] == team_name) | (matches_df['team2_name'] == team_name)) &
        (matches_df['date'] <= pd.Timestamp(target_date)) &
        (matches_df['date'] >= pd.Timestamp(target_date) - timedelta(days=window_days))
    ]

    if exclude_match_id is not None:
        recent = recent[recent['id'] != exclude_match_id]

    return len(recent)


def build_features(
    db_path: str = 'data/cs2_data.db',
    ratings_path: str = 'data/top_teams.txt',
    output_path: str = 'data/features.csv',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Строит признаки для всех карт из базы данных.

    Для каждой карты вычисляются признаки, используя только данные до даты этого матча.

    Returns:
        DataFrame с колонками:
        game_id, match_id, date, map_name, is_lan, y,
        delta_rating, form_diff_10, sos_form_diff_10,
        days_since_last_match_diff, recent_matches_count_diff,
        map_strength_diff, map_recent_form_diff,
        map_t_side_winrate_diff, map_ct_side_winrate_diff, map_sample_diff,
        rating_x_map_strength, form_x_map_form, rating_x_sos
    """
    if verbose:
        print("Загрузка данных из базы...")

    matches_df, games_df = load_data(db_path)
    ratings = load_team_ratings(ratings_path)

    if verbose:
        print(f"Загружено {len(matches_df)} матчей и {len(games_df)} карт")
        print(f"Загружено рейтингов для {len(ratings)} команд")

    # Merge games с matches для получения информации о командах и дате
    # match_id в games ссылается на id в matches
    df = games_df.merge(matches_df, left_on='match_id', right_on='id', how='left', suffixes=('', '_match'))

    # Сортируем по дате для правильного вычисления признаков
    df = df.sort_values('date').reset_index(drop=True)

    if verbose:
        print(f"Вычисление признаков для {len(df)} карт...")

    # Списки для признаков
    delta_ratings = []
    form_diffs = []
    sos_form_diffs = []
    days_since_diffs = []
    recent_matches_diffs = []
    map_strength_diffs = []
    map_recent_form_diffs = []
    map_t_winrate_diffs = []
    map_ct_winrate_diffs = []
    map_sample_diffs = []

    # Кэш для ускорения вычислений
    team_form_cache = {}
    team_map_cache = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), disable=not verbose):
        date = row['date'].to_pydatetime()
        map_name = row['map_name']
        team1 = row['team1_name']
        team2 = row['team2_name']
        match_id = row['match_id']

        # Рейтинг
        rating1 = get_team_rating(team1, ratings, 0)
        rating2 = get_team_rating(team2, ratings, 0)
        delta_ratings.append(rating1 - rating2)

        # Форма (за последние 10 карт)
        if team1 not in team_form_cache or team_form_cache[team1]['date'] != date:
            form1, sos1, days1 = compute_form_for_team(
                team1, matches_df, games_df, date, 10, ratings, exclude_match_id=match_id
            )
            team_form_cache[team1] = {'date': date, 'form': form1, 'sos': sos1, 'days': days1}
        else:
            form1 = team_form_cache[team1]['form']
            sos1 = team_form_cache[team1]['sos']
            days1 = team_form_cache[team1]['days']

        if team2 not in team_form_cache or team_form_cache[team2]['date'] != date:
            form2, sos2, days2 = compute_form_for_team(
                team2, matches_df, games_df, date, 10, ratings, exclude_match_id=match_id
            )
            team_form_cache[team2] = {'date': date, 'form': form2, 'sos': sos2, 'days': days2}
        else:
            form2 = team_form_cache[team2]['form']
            sos2 = team_form_cache[team2]['sos']
            days2 = team_form_cache[team2]['days']

        form_diffs.append(form1 - form2)
        sos_form_diffs.append(sos1 - sos2)
        days_since_diffs.append(min(days1 - days2, 30))  # capped at 30

        # Количество матчей за последние 14 дней
        matches1 = compute_recent_matches_count(team1, matches_df, date, 14, exclude_match_id=match_id)
        matches2 = compute_recent_matches_count(team2, matches_df, date, 14, exclude_match_id=match_id)
        recent_matches_diffs.append(matches1 - matches2)

        # Карточная статистика
        cache_key1 = (team1, map_name)
        cache_key2 = (team2, map_name)

        if cache_key1 not in team_map_cache or team_map_cache[cache_key1]['date'] != date:
            ms1, mrf1, mt1, mct1, mc1 = compute_map_stats_for_team(
                team1, map_name, matches_df, games_df, date, 365, exclude_match_id=match_id
            )
            team_map_cache[cache_key1] = {
                'date': date, 'ms': ms1, 'mrf': mrf1, 'mt': mt1, 'mct': mct1, 'mc': mc1
            }
        else:
            ms1 = team_map_cache[cache_key1]['ms']
            mrf1 = team_map_cache[cache_key1]['mrf']
            mt1 = team_map_cache[cache_key1]['mt']
            mct1 = team_map_cache[cache_key1]['mct']
            mc1 = team_map_cache[cache_key1]['mc']

        if cache_key2 not in team_map_cache or team_map_cache[cache_key2]['date'] != date:
            ms2, mrf2, mt2, mct2, mc2 = compute_map_stats_for_team(
                team2, map_name, matches_df, games_df, date, 365, exclude_match_id=match_id
            )
            team_map_cache[cache_key2] = {
                'date': date, 'ms': ms2, 'mrf': mrf2, 'mt': mt2, 'mct': mct2, 'mc': mc2
            }
        else:
            ms2 = team_map_cache[cache_key2]['ms']
            mrf2 = team_map_cache[cache_key2]['mrf']
            mt2 = team_map_cache[cache_key2]['mt']
            mct2 = team_map_cache[cache_key2]['mct']
            mc2 = team_map_cache[cache_key2]['mc']

        map_strength_diffs.append(ms1 - ms2)
        map_recent_form_diffs.append(mrf1 - mrf2)
        map_t_winrate_diffs.append(mt1 - mt2)
        map_ct_winrate_diffs.append(mct1 - mct2)

        # Логарифмическая разница количества карт
        map_sample_diffs.append(np.log(mc1 + 1) - np.log(mc2 + 1))

    # Добавляем признаки в DataFrame
    df['delta_rating'] = delta_ratings
    df['form_diff_10'] = form_diffs
    df['sos_form_diff_10'] = sos_form_diffs
    df['days_since_last_match_diff'] = days_since_diffs
    df['recent_matches_count_diff'] = recent_matches_diffs
    df['map_strength_diff'] = map_strength_diffs
    df['map_recent_form_diff'] = map_recent_form_diffs
    df['map_t_side_winrate_diff'] = map_t_winrate_diffs
    df['map_ct_side_winrate_diff'] = map_ct_winrate_diffs
    df['map_sample_diff'] = map_sample_diffs

    # Взаимодействия
    df['rating_x_map_strength'] = df['delta_rating'] * df['map_strength_diff']
    df['form_x_map_form'] = df['form_diff_10'] * df['map_recent_form_diff']
    df['rating_x_sos'] = df['delta_rating'] * df['sos_form_diff_10']

    # Заполняем NaN нулями
    feature_cols = [
        'delta_rating', 'form_diff_10', 'sos_form_diff_10',
        'days_since_last_match_diff', 'recent_matches_count_diff',
        'map_strength_diff', 'map_recent_form_diff',
        'map_t_side_winrate_diff', 'map_ct_side_winrate_diff', 'map_sample_diff',
        'rating_x_map_strength', 'form_x_map_form', 'rating_x_sos'
    ]
    df[feature_cols] = df[feature_cols].fillna(0)

    # Выбираем нужные колонки
    result = df[[
        'game_id', 'match_id', 'date', 'map_name', 'is_lan', 'y',
        'delta_rating', 'form_diff_10', 'sos_form_diff_10',
        'days_since_last_match_diff', 'recent_matches_count_diff',
        'map_strength_diff', 'map_recent_form_diff',
        'map_t_side_winrate_diff', 'map_ct_side_winrate_diff', 'map_sample_diff',
        'rating_x_map_strength', 'form_x_map_form', 'rating_x_sos'
    ]].copy()

    # Сохраняем в CSV
    result.to_csv(output_path, index=False)

    if verbose:
        print(f"Признаки сохранены в {output_path}")
        print(f"Итого строк: {len(result)}")
        print(f"Колонки: {list(result.columns)}")

    return result


if __name__ == '__main__':
    df = build_features()
    print("\nПервые 5 строк:")
    print(df.head())
    print("\nСтатистика:")
    print(df.describe())
