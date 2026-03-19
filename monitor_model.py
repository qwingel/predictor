"""
Скрипт для мониторинга качества модели
Отслеживает точность предсказаний и сохраняет историю
"""
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path

class ModelMonitor:
    def __init__(self, log_file='predictions_log.json', db_path='hltv_data.db'):
        self.log_file = log_file
        self.conn = sqlite3.connect(db_path)
        self.predictions = self._load_log()

    def _load_log(self):
        """Загрузить историю предсказаний"""
        if Path(self.log_file).exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []

    def _save_log(self):
        """Сохранить историю предсказаний"""
        with open(self.log_file, 'w') as f:
            json.dump(self.predictions, f, indent=2)

    def log_prediction(self, match_id, team1_id, team2_id, predicted_winner,
                      confidence, actual_winner=None):
        """
        Записать предсказание в лог

        Parameters:
        -----------
        match_id : int
            ID матча
        team1_id : int
            ID первой команды
        team2_id : int
            ID второй команды
        predicted_winner : str
            "Team 1" или "Team 2"
        confidence : float
            Уверенность модели (0-1)
        actual_winner : str, optional
            Реальный победитель (после матча)
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'match_id': match_id,
            'team1_id': team1_id,
            'team2_id': team2_id,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'actual_winner': actual_winner
        }

        self.predictions.append(entry)
        self._save_log()

        print(f"[OK] Предсказание записано (match_id={match_id})")

    def update_actual_result(self, match_id, actual_winner):
        """Обновить реальный результат матча"""
        for pred in self.predictions:
            if pred['match_id'] == match_id:
                pred['actual_winner'] = actual_winner
                self._save_log()
                print(f"[OK] Результат обновлен (match_id={match_id})")
                return True

        print(f"[ERROR] Предсказание не найдено (match_id={match_id})")
        return False

    def calculate_accuracy(self, days=None, last_n=None):
        """
        Рассчитать точность модели

        Parameters:
        -----------
        days : int, optional
            За последние N дней
        last_n : int, optional
            За последние N матчей
        """
        # Фильтровать предсказания с известным результатом
        completed = [p for p in self.predictions if p['actual_winner'] is not None]

        if len(completed) == 0:
            return None

        # Фильтровать по времени
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            completed = [p for p in completed
                        if datetime.fromisoformat(p['timestamp']) >= cutoff]

        # Взять последние N
        if last_n:
            completed = completed[-last_n:]

        if len(completed) == 0:
            return None

        # Рассчитать точность
        correct = sum(1 for p in completed
                     if p['predicted_winner'] == p['actual_winner'])

        accuracy = correct / len(completed)

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(completed),
            'period': f'last {days} days' if days else f'last {last_n} matches' if last_n else 'all time'
        }

    def get_statistics(self):
        """Получить детальную статистику"""
        completed = [p for p in self.predictions if p['actual_winner'] is not None]

        if len(completed) == 0:
            return {
                'total_predictions': len(self.predictions),
                'completed': 0,
                'pending': len(self.predictions)
            }

        correct = sum(1 for p in completed
                     if p['predicted_winner'] == p['actual_winner'])

        # Статистика по уверенности
        high_conf = [p for p in completed if p['confidence'] > 0.7]
        high_conf_correct = sum(1 for p in high_conf
                               if p['predicted_winner'] == p['actual_winner'])

        low_conf = [p for p in completed if p['confidence'] < 0.6]
        low_conf_correct = sum(1 for p in low_conf
                              if p['predicted_winner'] == p['actual_winner'])

        return {
            'total_predictions': len(self.predictions),
            'completed': len(completed),
            'pending': len(self.predictions) - len(completed),
            'overall_accuracy': correct / len(completed),
            'correct': correct,
            'incorrect': len(completed) - correct,
            'high_confidence': {
                'count': len(high_conf),
                'accuracy': high_conf_correct / len(high_conf) if high_conf else 0,
                'correct': high_conf_correct
            },
            'low_confidence': {
                'count': len(low_conf),
                'accuracy': low_conf_correct / len(low_conf) if low_conf else 0,
                'correct': low_conf_correct
            }
        }

    def print_report(self):
        """Вывести отчет о качестве модели"""
        print("\n" + "="*60)
        print("ОТЧЕТ О КАЧЕСТВЕ МОДЕЛИ")
        print("="*60)

        stats = self.get_statistics()

        print(f"\nВсего предсказаний: {stats['total_predictions']}")
        print(f"  Завершено: {stats['completed']}")
        print(f"  Ожидают результата: {stats['pending']}")

        if stats['completed'] > 0:
            print(f"\nОбщая точность: {stats['overall_accuracy']:.1%}")
            print(f"  Правильных: {stats['correct']}")
            print(f"  Неправильных: {stats['incorrect']}")

            print(f"\nВысокая уверенность (>70%):")
            print(f"  Матчей: {stats['high_confidence']['count']}")
            print(f"  Точность: {stats['high_confidence']['accuracy']:.1%}")

            print(f"\nНизкая уверенность (<60%):")
            print(f"  Матчей: {stats['low_confidence']['count']}")
            print(f"  Точность: {stats['low_confidence']['accuracy']:.1%}")

            # Точность за разные периоды
            print(f"\nТочность по периодам:")

            for period, kwargs in [
                ('Последние 7 дней', {'days': 7}),
                ('Последние 30 дней', {'days': 30}),
                ('Последние 100 матчей', {'last_n': 100})
            ]:
                acc = self.calculate_accuracy(**kwargs)
                if acc:
                    print(f"  {period}: {acc['accuracy']:.1%} ({acc['correct']}/{acc['total']})")

            # Проверка на drift
            print(f"\n" + "-"*60)
            recent = self.calculate_accuracy(days=30)
            overall = stats['overall_accuracy']

            if recent and abs(recent['accuracy'] - overall) > 0.05:
                print("[WARNING] ВНИМАНИЕ: Обнаружен drift!")
                print(f"   Общая точность: {overall:.1%}")
                print(f"   Последние 30 дней: {recent['accuracy']:.1%}")
                print(f"   Разница: {(recent['accuracy'] - overall)*100:+.1f}%")
                print("\n   Рекомендация: Рассмотрите переобучение модели")
            else:
                print("[OK] Модель стабильна, drift не обнаружен")

        print("\n" + "="*60)

    def close(self):
        """Закрыть соединение"""
        self.conn.close()


# Пример использования
if __name__ == '__main__':
    monitor = ModelMonitor()

    # Пример 1: Записать предсказание
    print("\n### ПРИМЕР 1: Запись предсказания ###")
    monitor.log_prediction(
        match_id=12345,
        team1_id=67,
        team2_id=110,
        predicted_winner="Team 1",
        confidence=0.653
    )

    # Пример 2: Обновить результат
    print("\n### ПРИМЕР 2: Обновление результата ###")
    monitor.update_actual_result(
        match_id=12345,
        actual_winner="Team 1"  # Модель была права!
    )

    # Пример 3: Посмотреть статистику
    print("\n### ПРИМЕР 3: Статистика ###")
    monitor.print_report()

    monitor.close()
