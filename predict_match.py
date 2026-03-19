"""
Простой скрипт для предсказания исходов матчей
Использует baseline модель (62.27% точности)
"""
import pickle
import pandas as pd
import sqlite3
from datetime import datetime

class MatchPredictor:
    def __init__(self, model_path='best_model_logistic.pkl', db_path='hltv_data.db'):
        """Инициализация предиктора"""
        # Загрузить модель
        try:
            with open(model_path, 'rb') as f:
                self.model, _ = pickle.load(f)
            print(f"[OK] Модель загружена: {model_path}")
        except FileNotFoundError:
            print(f"[ERROR] Модель не найдена: {model_path}")
            print("  Используйте model.pkl или final_model.pkl")
            raise

        # Определить порядок признаков
        self.feature_names = ['delta_rating', 'map_missing_flag', 'LAN_flag', 'h2h_shrunk', 'h2h_count']

        # Подключиться к БД
        self.conn = sqlite3.connect(db_path)
        print(f"[OK] База данных подключена: {db_path}")

    def get_team_rating(self, team_id, as_of_date=None):
        """Получить рейтинг команды на дату"""
        if as_of_date is None:
            as_of_date = datetime.now().strftime('%Y-%m-%d')

        query = """
        SELECT rank_position
        FROM team_rankings_daily
        WHERE team_id = ? AND ranking_date <= ?
        ORDER BY ranking_date DESC
        LIMIT 1
        """
        result = pd.read_sql_query(query, self.conn, params=(team_id, as_of_date))

        if len(result) == 0:
            return None

        return result['rank_position'].iloc[0]

    def get_h2h_stats(self, team1_id, team2_id, limit=10):
        """Получить статистику личных встреч"""
        query = """
        SELECT winner_team_id
        FROM matches
        WHERE (team1_id = ? AND team2_id = ?)
           OR (team1_id = ? AND team2_id = ?)
        ORDER BY start_time DESC
        LIMIT ?
        """
        result = pd.read_sql_query(
            query, self.conn,
            params=(team1_id, team2_id, team2_id, team1_id, limit)
        )

        if len(result) == 0:
            return 0, 0  # h2h_shrunk, h2h_count

        # Подсчитать победы
        team1_wins = (result['winner_team_id'] == team1_id).sum()
        team2_wins = (result['winner_team_id'] == team2_id).sum()

        h2h_count = len(result)
        h2h_shrunk = team2_wins - team1_wins  # положительное = Team2 выигрывает чаще

        return h2h_shrunk, h2h_count

    def predict(self, team1_id, team2_id, is_lan=0, map_name=None):
        """
        Предсказать исход матча

        Parameters:
        -----------
        team1_id : int
            ID первой команды
        team2_id : int
            ID второй команды
        is_lan : int
            0 = online, 1 = LAN
        map_name : str, optional
            Название карты (если известно)

        Returns:
        --------
        dict : результат предсказания
        """
        # 1. Получить рейтинги
        rating1 = self.get_team_rating(team1_id)
        rating2 = self.get_team_rating(team2_id)

        if rating1 is None or rating2 is None:
            return {
                'error': 'Команда не найдена в рейтинге',
                'team1_rating': rating1,
                'team2_rating': rating2
            }

        delta_rating = rating2 - rating1

        # 2. Получить h2h
        h2h_shrunk, h2h_count = self.get_h2h_stats(team1_id, team2_id)

        # 3. Map missing flag
        map_missing_flag = 1 if map_name is None else 0

        # 4. Подготовить признаки
        features_dict = {
            'delta_rating': delta_rating,
            'map_missing_flag': map_missing_flag,
            'LAN_flag': is_lan,
            'h2h_shrunk': h2h_shrunk,
            'h2h_count': h2h_count
        }

        X = pd.DataFrame([features_dict])[self.feature_names]

        # 5. Предсказать
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # 6. Форматировать результат
        winner = "Team 1" if prediction == -1 else "Team 2"
        winner_id = team1_id if prediction == -1 else team2_id
        confidence = max(probabilities)

        result = {
            'winner': winner,
            'winner_id': winner_id,
            'confidence': confidence,
            'probabilities': {
                'team1': probabilities[0],
                'team2': probabilities[1]
            },
            'features': features_dict,
            'ratings': {
                'team1': rating1,
                'team2': rating2,
                'delta': delta_rating
            },
            'h2h': {
                'shrunk': h2h_shrunk,
                'count': h2h_count
            }
        }

        return result

    def predict_and_print(self, team1_id, team2_id, team1_name=None, team2_name=None,
                         is_lan=0, map_name=None):
        """Предсказать и красиво вывести результат"""
        print("\n" + "="*60)
        print("ПРЕДСКАЗАНИЕ ИСХОДА МАТЧА")
        print("="*60)

        # Получить названия команд из БД если не указаны
        if team1_name is None:
            query = "SELECT team_name FROM teams WHERE team_id = ?"
            result = pd.read_sql_query(query, self.conn, params=(team1_id,))
            team1_name = result['team_name'].iloc[0] if len(result) > 0 else f"Team {team1_id}"

        if team2_name is None:
            query = "SELECT team_name FROM teams WHERE team_id = ?"
            result = pd.read_sql_query(query, self.conn, params=(team2_id,))
            team2_name = result['team_name'].iloc[0] if len(result) > 0 else f"Team {team2_id}"

        print(f"\n{team1_name} vs {team2_name}")
        print(f"Формат: {'LAN' if is_lan else 'Online'}")
        if map_name:
            print(f"Карта: {map_name}")

        # Предсказать
        result = self.predict(team1_id, team2_id, is_lan, map_name)

        if 'error' in result:
            print(f"\n[ERROR] Ошибка: {result['error']}")
            return result

        # Вывести результат
        print("\n" + "-"*60)
        print("РЕЗУЛЬТАТ:")
        print("-"*60)

        winner_name = team1_name if result['winner'] == "Team 1" else team2_name
        print(f"\n>>> Победитель: {winner_name}")
        print(f"   Уверенность: {result['confidence']:.1%}")

        print(f"\nВероятности:")
        print(f"  {team1_name}: {result['probabilities']['team1']:.1%}")
        print(f"  {team2_name}: {result['probabilities']['team2']:.1%}")

        print(f"\nРейтинги:")
        print(f"  {team1_name}: #{result['ratings']['team1']}")
        print(f"  {team2_name}: #{result['ratings']['team2']}")
        print(f"  Разница: {result['ratings']['delta']:+d}")

        if result['h2h']['count'] > 0:
            print(f"\nЛичные встречи (последние {result['h2h']['count']}):")
            if result['h2h']['shrunk'] > 0:
                print(f"  {team2_name} выигрывает чаще")
            elif result['h2h']['shrunk'] < 0:
                print(f"  {team1_name} выигрывает чаще")
            else:
                print(f"  Равный счет")

        print("\n" + "="*60)

        return result

    def close(self):
        """Закрыть соединение с БД"""
        self.conn.close()


# Пример использования
if __name__ == '__main__':
    # Создать предиктор
    predictor = MatchPredictor()

    # Пример 1: Предсказание с ID команд
    print("\n### ПРИМЕР 1: Предсказание по ID ###")
    result = predictor.predict_and_print(
        team1_id=67,    # Замените на реальные ID
        team2_id=110,
        is_lan=1
    )

    # Пример 2: Предсказание с названиями
    print("\n### ПРИМЕР 2: С названиями команд ###")
    result = predictor.predict_and_print(
        team1_id=67,
        team2_id=110,
        team1_name="Natus Vincere",
        team2_name="FaZe Clan",
        is_lan=1,
        map_name="Mirage"
    )

    # Пример 3: Программное использование
    print("\n### ПРИМЕР 3: Программное использование ###")
    result = predictor.predict(team1_id=67, team2_id=110, is_lan=1)

    if 'error' not in result:
        print(f"Победитель: {result['winner']}")
        print(f"Уверенность: {result['confidence']:.1%}")
        print(f"Признаки: {result['features']}")

    # Закрыть
    predictor.close()
