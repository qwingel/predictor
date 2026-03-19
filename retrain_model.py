"""
Скрипт для переобучения модели
Использовать раз в 3-6 месяцев или при падении точности
"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from datetime import datetime, timedelta

def retrain_model(db_path='hltv_data.db',
                 start_date=None,
                 end_date=None,
                 output_path='model_retrained.pkl'):
    """
    Переобучить модель на свежих данных

    Parameters:
    -----------
    db_path : str
        Путь к базе данных
    start_date : str, optional
        Начальная дата (YYYY-MM-DD)
    end_date : str, optional
        Конечная дата (YYYY-MM-DD)
    output_path : str
        Путь для сохранения новой модели
    """
    print("="*80)
    print("ПЕРЕОБУЧЕНИЕ МОДЕЛИ")
    print("="*80)

    conn = sqlite3.connect(db_path)

    # 1. Загрузить данные
    print("\n[1/7] Загрузка данных...")

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"Период: {start_date} до {end_date}")

    # Загрузить training examples
    query = """
    SELECT
        te.id,
        te.team1_id,
        te.team2_id,
        te.as_of_time,
        te.y,
        m.best_of,
        m.lan_flag
    FROM training_examples te
    JOIN matches m ON te.match_id = m.match_id
    WHERE te.as_of_time >= ? AND te.as_of_time <= ?
    ORDER BY te.as_of_time
    """

    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    print(f"Загружено {len(df)} примеров")

    if len(df) < 1000:
        print("[WARNING] ВНИМАНИЕ: Мало данных для переобучения!")
        print("   Рекомендуется минимум 1000 примеров")
        return None

    # 2. Построить признаки
    print("\n[2/7] Построение признаков...")

    def get_team_rating(team_id, as_of_date):
        query = """
        SELECT rank_position
        FROM team_rankings_daily
        WHERE team_id = ? AND ranking_date <= ?
        ORDER BY ranking_date DESC
        LIMIT 1
        """
        result = pd.read_sql_query(query, conn, params=(team_id, as_of_date))
        return result['rank_position'].iloc[0] if len(result) > 0 else 150

    def get_h2h(team1_id, team2_id, as_of_date):
        query = """
        SELECT winner_team_id
        FROM matches
        WHERE ((team1_id = ? AND team2_id = ?) OR (team1_id = ? AND team2_id = ?))
            AND start_time < ?
        ORDER BY start_time DESC
        LIMIT 10
        """
        result = pd.read_sql_query(
            query, conn,
            params=(team1_id, team2_id, team2_id, team1_id, as_of_date)
        )

        if len(result) == 0:
            return 0, 0

        team1_wins = (result['winner_team_id'] == team1_id).sum()
        team2_wins = (result['winner_team_id'] == team2_id).sum()

        return team2_wins - team1_wins, len(result)

    features_list = []

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"  Прогресс: {idx}/{len(df)}")

        rating1 = get_team_rating(row['team1_id'], row['as_of_time'])
        rating2 = get_team_rating(row['team2_id'], row['as_of_time'])

        h2h_shrunk, h2h_count = get_h2h(
            row['team1_id'], row['team2_id'], row['as_of_time']
        )

        features_list.append({
            'delta_rating': rating2 - rating1,
            'map_missing_flag': 0,  # Упрощение
            'LAN_flag': row['lan_flag'],
            'h2h_shrunk': h2h_shrunk,
            'h2h_count': h2h_count,
            'y': row['y']
        })

    df_features = pd.DataFrame(features_list)
    print(f"Построено {len(df_features)} feature vectors")

    # 3. Temporal split
    print("\n[3/7] Temporal train/test split...")

    split_idx = int(len(df_features) * 0.8)
    df_train = df_features.iloc[:split_idx]
    df_test = df_features.iloc[split_idx:]

    print(f"Train: {len(df_train)} примеров")
    print(f"Test: {len(df_test)} примеров")

    feature_cols = ['delta_rating', 'map_missing_flag', 'LAN_flag', 'h2h_shrunk', 'h2h_count']

    X_train = df_train[feature_cols].values
    y_train = df_train['y'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['y'].values

    # 4. Scale
    print("\n[4/7] Масштабирование признаков...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train
    print("\n[5/7] Обучение модели...")
    model = LogisticRegression(C=0.5, max_iter=2000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 6. Evaluate
    print("\n[6/7] Оценка качества...")

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\nTrain accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Team 1', 'Team 2']))

    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\nConfusion Matrix:\n{cm}")

    # 7. Save
    print("\n[7/7] Сохранение модели...")

    baseline_acc = 0.6227

    if test_acc >= baseline_acc - 0.02:  # Не хуже baseline на 2%
        with open(output_path, 'wb') as f:
            pickle.dump((model, feature_cols), f)

        print(f"[OK] Модель сохранена: {output_path}")
        print(f"  Точность: {test_acc*100:.2f}%")

        if test_acc > baseline_acc:
            print(f"  [OK] УЛУЧШЕНИЕ: +{(test_acc - baseline_acc)*100:.2f}% над baseline")
        else:
            print(f"  ≈ Похожа на baseline ({baseline_acc*100:.2f}%)")

        return {
            'success': True,
            'test_accuracy': test_acc,
            'train_accuracy': train_acc,
            'model_path': output_path
        }
    else:
        print(f"[ERROR] Модель НЕ сохранена (хуже baseline)")
        print(f"  Baseline: {baseline_acc*100:.2f}%")
        print(f"  Новая модель: {test_acc*100:.2f}%")
        print(f"  Разница: {(test_acc - baseline_acc)*100:.2f}%")

        return {
            'success': False,
            'test_accuracy': test_acc,
            'reason': 'Worse than baseline'
        }

    conn.close()


# Пример использования
if __name__ == '__main__':
    print("\n### ПЕРЕОБУЧЕНИЕ МОДЕЛИ ###\n")

    # Переобучить на данных за последний год
    result = retrain_model(
        start_date='2023-01-01',
        end_date='2024-12-31',
        output_path='model_retrained.pkl'
    )

    if result and result['success']:
        print("\n" + "="*80)
        print("ПЕРЕОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
        print("="*80)
        print(f"\nНовая модель: {result['model_path']}")
        print(f"Точность: {result['test_accuracy']*100:.2f}%")
        print("\nЧтобы использовать новую модель:")
        print("  1. Протестируйте её на реальных данных")
        print("  2. Если работает хорошо - замените best_model_logistic.pkl")
        print("  3. Обновите predict_match.py для использования новой модели")
    else:
        print("\n" + "="*80)
        print("ПЕРЕОБУЧЕНИЕ НЕ УДАЛОСЬ")
        print("="*80)
        print("\nНовая модель хуже baseline, продолжайте использовать старую.")
