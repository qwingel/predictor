"""
Простой скрипт для предсказаний по названиям команд
"""
import sqlite3
import pandas as pd
from predict_match import MatchPredictor

class SimplePredictor:
    def __init__(self):
        self.predictor = MatchPredictor()
        self.conn = sqlite3.connect('hltv_data.db')

    def find_team(self, team_name):
        """Найти команду по названию"""
        query = """
        SELECT team_id, team_name
        FROM teams
        WHERE LOWER(team_name) LIKE LOWER(?)
        ORDER BY team_name
        """
        result = pd.read_sql_query(query, self.conn, params=(f'%{team_name}%',))
        return result

    def predict_by_names(self, team1_name, team2_name, is_lan=0, map_name=None):
        """
        Предсказать матч по названиям команд

        Parameters:
        -----------
        team1_name : str
            Название первой команды (или часть названия)
        team2_name : str
            Название второй команды (или часть названия)
        is_lan : int
            0 = online, 1 = LAN
        map_name : str, optional
            Название карты (Dust2, Mirage, Inferno, Nuke, Overpass, Vertigo, Ancient, Anubis)

        Returns:
        --------
        dict : результат предсказания
        """
        # Найти команды
        teams1 = self.find_team(team1_name)
        teams2 = self.find_team(team2_name)

        if len(teams1) == 0:
            print(f"[ERROR] Команда '{team1_name}' не найдена")
            print("Попробуйте другое название или часть названия")
            return None

        if len(teams2) == 0:
            print(f"[ERROR] Команда '{team2_name}' не найдена")
            print("Попробуйте другое название или часть названия")
            return None

        # Если найдено несколько - показать варианты
        if len(teams1) > 1:
            print(f"\n[INFO] Найдено несколько команд для '{team1_name}':")
            print(teams1.to_string(index=False))
            print(f"\nИспользую первую: {teams1.iloc[0]['team_name']}")

        if len(teams2) > 1:
            print(f"\n[INFO] Найдено несколько команд для '{team2_name}':")
            print(teams2.to_string(index=False))
            print(f"\nИспользую первую: {teams2.iloc[0]['team_name']}")

        # Взять первую найденную команду
        team1_id = int(teams1.iloc[0]['team_id'])
        team1_full_name = teams1.iloc[0]['team_name']
        team2_id = int(teams2.iloc[0]['team_id'])
        team2_full_name = teams2.iloc[0]['team_name']

        # Предсказать
        result = self.predictor.predict_and_print(
            team1_id=team1_id,
            team2_id=team2_id,
            team1_name=team1_full_name,
            team2_name=team2_full_name,
            is_lan=is_lan,
            map_name=map_name
        )

        return result

    def predict_bo3(self, team1_name, team2_name, maps, is_lan=1):
        """
        Предсказать BO3 матч (3 карты)

        Parameters:
        -----------
        team1_name : str
            Название первой команды
        team2_name : str
            Название второй команды
        maps : list
            Список карт, например ['Dust2', 'Mirage', 'Inferno']
        is_lan : int
            0 = online, 1 = LAN
        """
        print("\n" + "="*60)
        print(f"ПРЕДСКАЗАНИЕ BO3: {team1_name} vs {team2_name}")
        print("="*60)

        results = []
        team1_wins = 0
        team2_wins = 0

        for i, map_name in enumerate(maps, 1):
            print(f"\n--- КАРТА {i}: {map_name} ---")
            result = self.predict_by_names(team1_name, team2_name, is_lan, map_name)

            if result and 'error' not in result:
                results.append(result)
                if result['winner'] == 'Team 1':
                    team1_wins += 1
                else:
                    team2_wins += 1

        # Итоговый результат
        print("\n" + "="*60)
        print("ИТОГОВЫЙ РЕЗУЛЬТАТ BO3")
        print("="*60)
        print(f"\nСчет по картам: {team1_wins} - {team2_wins}")

        if team1_wins > team2_wins:
            print(f"Победитель серии: {team1_name}")
        else:
            print(f"Победитель серии: {team2_name}")

        print("\n" + "="*60)

        return results

    def close(self):
        """Закрыть соединения"""
        self.predictor.close()
        self.conn.close()


# Примеры использования
if __name__ == '__main__':
    predictor = SimplePredictor()

    print("\n### ПРИМЕР 1: Простое предсказание ###")
    # Предсказать один матч
    result = predictor.predict_by_names(
        team1_name="Natus Vincere",
        team2_name="FaZe",
        is_lan=1,
        map_name="Mirage"
    )

    print("\n### ПРИМЕР 2: BO3 матч ###")
    # Предсказать BO3 (3 карты)
    results = predictor.predict_bo3(
        team1_name="Natus Vincere",
        team2_name="FaZe",
        maps=["Dust2", "Mirage", "Inferno"],
        is_lan=1
    )

    predictor.close()
