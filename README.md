# Система предсказания исхода карт CS2

Машинное обучение для предсказания победителя карты в киберспортивных матчах Counter-Strike 2.

## Описание

Проект представляет собой симметричную модель на основе **LightGBM** с `boosting_type='goss'` для предсказания исхода конкретной карты в матче CS2. Модель учитывает форму команд, статистику на карте, рейтинги и другие факторы.

Поддерживает:
- Предсказание для одной карты
- Предсказание для всех карт матча (`predict_for_all_maps`)
- Рекомендации по ставкам с расчётом EV и критерием Келли (`bet_recommendation`)
- Пост-матч анализ через Telegram-модуль

## Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск полного пайплайна

```bash
cd train
python main.py
```

Пайплайн автоматически выполнит:
1. Построение признаков из базы данных
2. Обучение модели LightGBM
3. Оценку качества модели
4. Сохранение результатов

## Требования к данным

Для работы необходимы следующие файлы:

| Файл | Описание |
|------|----------|
| `data/cs2_data.db` | SQLite база с матчами и картами |
| `data/top_teams.txt` | Рейтинги команд (формат HLTV) |

### Формат базы данных

**Таблица `matches`:**
```sql
id, hltv_match_id, date, tournament, is_lan, team1_name, team2_name, team1_score, team2_score, winner
```

**Таблица `games`:**
```sql
id, match_id, map_name, team1_score, team2_score, y
```

### Формат рейтингов команд

```
1: "Vitality", 1000
2: "Furia", 561
3: "NAVI", 489
```

## Использование

### Предсказание для конкретного матча

```python
from predict import predict_match, load_model

# Загрузка модели
model = load_model('models/model_lgbm.pkl')

# Предсказание
result = predict_match(
    model,
    team1='Vitality',
    team2='Furia',
    map_name='Inferno',
    is_lan=1  # LAN турнир
)

print(f"Вероятность победы: {max(result['prob_team1'], result['prob_team2']):.2%}")
```

### Предсказание для всех карт матча

```python
from predict import predict_for_all_maps

result = predict_for_all_maps(model, 'G2', 'GamerLegion', is_lan=1)

for map_name, (winner, pred, p1, p2) in result.items():
    print(f"{map_name}: {winner}, {p1:.2%} vs {p2:.2%}")
```

### Рекомендации по ставкам

```python
from bet_recommendation import evaluate_bet

# Оценить ставку: модель дала 65%, букмекер даёт коэффициент 1.83
result = evaluate_bet(model_prob=0.65, bookmaker_odds=1.83)

print(result['recommendation'])   # "СТАВИТЬ" / "НЕ СТАВИТЬ" / "РИСК"
print(result['bet_size_percent']) # размер ставки по Келли (доля от банка)
print(result['ev'])               # математическое ожидание
print(result['edge'])             # преимущество над линией
print(result['fair_odds'])        # "справедливый" коэффициент модели
```

### Примеры предсказаний

```python
# Онлайн матч
result = predict_match(model, 'NAVI', 'FaZe', 'Mirage', is_lan=0)

# LAN турнир
result = predict_match(model, 'Vitality', 'Spirit', 'Ancient', is_lan=1)
```

### Пост-матч анализ (Telegram)

Скрипт `telegram/reader_after_event.py` читает пары команд и карт из `after_event.txt` и делает предсказания.

Формат `after_event.txt`:
```
Vitality,Furia,Inferno
NAVI,G2,Mirage
```

```bash
cd telegram
python reader_after_event.py
```

## Архитектура

### Признаки

Все признаки симметричные (`delta = team1_value - team2_value`):

| Категория | Признаки |
|-----------|----------|
| **Базовые** | `delta_rating`, `form_diff_10`, `sos_form_diff_10` |
| **Временные** | `days_since_last_match_diff`, `recent_matches_count_diff` |
| **Карточные** | `map_strength_diff`, `map_recent_form_diff`, `map_t_side_winrate_diff`, `map_ct_side_winrate_diff`, `map_sample_diff` |
| **Взаимодействия** | `rating_x_map_strength`, `form_x_map_form`, `rating_x_sos` |

### Симметризация предсказаний

Модель использует двойной проход для гарантированной симметричности:
1. Прямой проход: `P(team1 | team1, team2)`
2. Обратный проход: `P(team2 | team2, team1)` (с инвертированными признаками)
3. Усреднение: `P_sym = (P_AB + (1 - P_BA)) / 2`
4. Temperature scaling для калибровки

Это гарантирует: `P(team1, team2) + P(team2, team1) = 1`

### Веса матчей

При обучении более свежие матчи имеют больший вес:

| Давность | Вес |
|----------|-----|
| 0-90 дней | 1.0 |
| 90-180 дней | 0.7 |
| 180-365 дней | 0.3 |
| > 365 дней | 0.0 |

### Временной сплит данных

Разделение производится по уникальным датам (все матчи одной даты в одной выборке):

| Набор | Доля | Период |
|-------|------|--------|
| Train | 70% | 2025-03-24 - 2025-11-20 |
| Validation | 15% | 2025-11-21 - 2026-01-31 |
| Test | 15% | 2026-02-02 - 2026-03-24 |

## Результаты модели

### Метрики на тестовой выборке (после ретрейна, 2026-04-02)

| Метрика | Значение | Цель | Статус |
|---------|----------|------|--------|
| **ROC-AUC** | **0.7020** | > 0.68 | ДОСТИГНУТ |
| Brier Score | 0.2198 | - | - |
| Accuracy | 65.69% | - | - |

### Параметры модели

```python
{
    'boosting_type': 'goss',
    'n_estimators': 900,
    'learning_rate': 0.01,
    'num_leaves': 18,
    'max_depth': 4,
    'reg_lambda': 15,
    'reg_alpha': 1.5,
    'min_child_samples': 40,
    'colsample_bytree': 0.78,
    'random_state': 42
}
```

## Структура проекта

```
predictor/
├── predict.py                  # Предсказания + predict_for_all_maps
├── bet_recommendation.py       # Рекомендации по ставкам (EV, Kelly)
├── validate_model.py           # Валидация модели
├── requirements.txt            # Зависимости Python
├── README.md                   # Этот файл
│
├── train/
│   ├── main.py                 # Точка входа, полный пайплайн
│   ├── features.py             # Построение признаков
│   ├── train.py                # Обучение модели
│   ├── train_with_all_matches.py # Обучение на всех матчах
│   ├── evaluate.py             # Оценка качества
│   ├── utils.py                # Утилиты (нормализация, веса)
│   └── normalize_data.py       # Нормализация данных
│
├── data/
│   ├── cs2_data.db             # База данных матчей
│   ├── top_teams.txt           # Рейтинги команд
│   ├── features.csv            # Построенные признаки
│   ├── train_report.txt        # Отчёт об обучении
│   ├── final_train_report.txt  # Финальный отчёт об обучении
│   └── evaluate_report.txt     # Отчёт об оценке
│
├── models/
│   ├── model_lgbm.pkl          # Обученная модель
│   └── model_lgbm_final.pkl    # Финальная модель (ретрейн)
│
├── png/
│   ├── calibration_curve.png   # Калибровочная кривая
│   └── predictions_distribution.png  # Распределение предсказаний
│
└── telegram/
    ├── after_event.txt         # Входные данные для пост-анализа
    └── reader_after_event.py   # Скрипт пост-матч анализа
```

## Модули

### train/main.py
Запускает полный пайплайн: построение признаков -> обучение -> оценка.

### train/features.py
Извлекает данные из SQLite, вычисляет признаки с использованием скользящих окон и кэширования.

### train/train.py
Обучает LightGBM модель с временным сплитом и весами матчей. Сохраняет модель и отчёт.

### train/train_with_all_matches.py
Альтернативный скрипт обучения на всех доступных матчах (без разбиения на train/test).

### train/evaluate.py
Оценивает качество модели: ROC-AUC, Brier Score, Accuracy, проверка симметричности, калибровочная кривая.

### train/utils.py
Вспомогательные функции:
- `normalize_team_name()` - нормализация названий команд (lowercase + маппинг)
- `load_team_ratings()` - загрузка рейтингов из файла
- `get_team_rating()` - получение рейтинга команды с fallback
- `step_weight()` - ступенчатая схема весов по давности
- `rolling_weighted_mean()` - взвешенное скользящее среднее

### predict.py
API для предсказаний:
- `predict_match()` - предсказание для одной карты с симметризацией и temperature scaling
- `predict_for_all_maps()` - предсказание для всех 7 карт CS2 за один вызов
- `predict_match_swapped()` - проверка симметричности предсказаний

### bet_recommendation.py
Расчёт рекомендаций по ставкам:
- `evaluate_bet()` - полная оценка ставки (EV, edge, Kelly, confidence)
- `calculate_ev()` - математическое ожидание ставки
- `calculate_kelly_size()` - оптимальный размер ставки по критерию Келли (25% fractional)
- `calculate_confidence_interval()` - доверительный интервал вероятности

Логика принятия решений:
| Условие | Рекомендация |
|---------|-------------|
| EV <= 0 | НЕ СТАВИТЬ |
| EV > 0, но edge < 3% | НЕ СТАВИТЬ |
| EV > 0, edge >= 3%, EV_lower < 0 | РИСК |
| EV > 0, edge >= 3%, EV_lower >= 0 | СТАВИТЬ |

### validate_model.py
Валидация модели: проверка на утечку данных, стабильность признаков, корректность симметризации.

### telegram/reader_after_event.py
Скрипт пост-матч анализа. Читает пары команд из `after_event.txt` (формат: `team1,team2,map`), делает предсказания и сохраняет результаты.

## Зависимости

```
pandas>=2.0.0
numpy>=1.24.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
joblib>=1.3.0
tqdm>=4.65.0
```

## Особенности

1. **Нормализация имён команд**: Все названия автоматически приводятся к lowercase с применением маппинга известных вариаций (NAVI -> natus vincere, Furia -> furia, и т.д.)

2. **Симметричность модели**: При перестановке команд местами предсказание инвертируется: `P(team1, team2) = 1 - P(team2, team1)`. Достигается двойным проходом с инвертированными признаками.

3. **Отсутствие утечки данных**: Идентификаторы и целевая переменная не входят в признаки, при расчёте используется фильтр по дате.

4. **Кэширование вычислений**: Для ускорения используется кэш признаков команд.

5. **Temperature scaling**: Калибровка вероятностей после симметризации для улучшения качества предсказаний.

6. **Kelly criterion**: Оптимальный размер ставки рассчитывается по дробному Келли (25%) с ограничением 1-5% от банка.

## Лицензия

MIT
