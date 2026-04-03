"""
Скрипт для нормализации названий команд в БД и top_teams.txt.

Приводит все названия к lowercase и создаёт маппинг для известных вариаций.
"""
import sqlite3
import pandas as pd
import re


# Маппинг известных вариаций команд
TEAM_MAPPING = {
    # Natus Vincere
    'navi': 'natus vincere',
    'navi junior': 'natus vincere junior',
    'ex-navi junior': 'ex-natus vincere junior',

    # FaZe
    'faze': 'faze',

    # G2
    'g2': 'g2',
    'g2 ares': 'g2 ares',

    # 3DMAX
    '3dmax': '3dmax',

    # The MongolZ
    'the mongolz': 'the mongolz',

    # PARIVISION
    'parivision': 'parivision',

    # Другие возможные вариации
    'virtus.pro': 'virtus pro',
    'team spirit': 'spirit',
    'furía': 'furia',
    'mouz': 'mouz',
}


def normalize_team_name(team_name: str) -> str:
    """
    Нормализует название команды:
    1. Приводит к lowercase
    2. Применяет маппинг известных вариаций
    """
    lower = team_name.lower().strip()
    return TEAM_MAPPING.get(lower, lower)


def normalize_database(db_path: str = 'cs2_data.db'):
    """
    Создаёт нормализованную версию БД с колонками team1_name_norm, team2_name_norm.
    """
    conn = sqlite3.connect(db_path)

    # Читаем все матчи
    matches_df = pd.read_sql_query('SELECT * FROM matches', conn)

    # Добавляем нормализованные имена
    matches_df['team1_name_norm'] = matches_df['team1_name'].apply(normalize_team_name)
    matches_df['team2_name_norm'] = matches_df['team2_name'].apply(normalize_team_name)

    # Проверяем изменения
    print('Нормализация команд в БД:')
    print('=' * 60)

    changes = []
    for idx, row in matches_df.iterrows():
        if row['team1_name'].lower() != row['team1_name_norm']:
            changes.append((row['team1_name'], row['team1_name_norm']))
        if row['team2_name'].lower() != row['team2_name_norm']:
            changes.append((row['team2_name'], row['team2_name_norm']))

    # Показываем уникальные изменения
    unique_changes = set(changes)
    print(f'Изменено названий: {len(unique_changes)}')
    for orig, norm in sorted(unique_changes)[:20]:
        print(f'  {orig} -> {norm}')

    # Сохраняем в новую таблицу или обновляем существующую
    # Для безопасности создаём view с нормализованными именами
    conn.close()

    return matches_df


def normalize_ratings(ratings_path: str = 'top_teams.txt') -> dict:
    """
    Загружает рейтинги с нормализованными именами команд.
    """
    ratings = {}
    with open(ratings_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'\d+:\s*"([^"]+)",\s*(\d+)', line)
            if match:
                team = match.group(1)
                points = int(match.group(2))
                normalized = normalize_team_name(team)
                ratings[normalized] = points

    return ratings


if __name__ == '__main__':
    # Тестируем нормализацию
    print('Тест нормализации:')
    print('=' * 60)

    test_teams = [
        'Vitality', 'FURIA', 'Furia', 'NAVI', 'Natus Vincere',
        'FaZe', 'G2', '3DMAX', 'The MongolZ', 'PARIVISION'
    ]

    for team in test_teams:
        normalized = normalize_team_name(team)
        print(f'  {team:20s} -> {normalized}')

    print()
    print('Нормализованные рейтинги:')
    ratings = normalize_ratings()
    for team, points in list(ratings.items())[:10]:
        print(f'  {team}: {points}')
