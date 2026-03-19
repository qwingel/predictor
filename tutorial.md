# Как использовать систему предсказаний

## Быстрый старт

### Вариант 1: Простое предсказание (рекомендуется)

Используй `simple_predict.py` - передаешь названия команд и карту, получаешь предсказание.

```python
from simple_predict import SimplePredictor

predictor = SimplePredictor()

# Предсказать один матч
result = predictor.predict_by_names(
    team1_name="Natus Vincere",  # Название первой команды
    team2_name="FaZe",            # Название второй команды
    is_lan=1,                     # 1 = LAN, 0 = Online
    map_name="Mirage"             # Название карты (опционально)
)

predictor.close()
```

### Вариант 2: BO3 матч (3 карты)

```python
from simple_predict import SimplePredictor

predictor = SimplePredictor()

# Предсказать BO3
results = predictor.predict_bo3(
    team1_name="Natus Vincere",
    team2_name="FaZe",
    maps=["Dust2", "Mirage", "Inferno"],  # Список карт
    is_lan=1
)

predictor.close()
```

---

## Параметры

### team1_name, team2_name (обязательно)
- Название команды или часть названия
- Примеры: "Natus Vincere", "FaZe", "G2", "Vitality"
- Можно писать частично: "Natus" найдет "Natus Vincere"

### is_lan (обязательно)
- `1` = LAN турнир
- `0` = Online матч

### map_name (опционально)
- Название карты
- Варианты: `"Dust2"`, `"Mirage"`, `"Inferno"`, `"Nuke"`, `"Overpass"`, `"Vertigo"`, `"Ancient"`, `"Anubis"`
- Если не указать - модель будет считать что карта неизвестна

---

## Что возвращает

Результат содержит:

```python
{
    'winner': 'Team 1',              # Победитель
    'winner_id': 4,                  # ID победителя
    'confidence': 0.61,              # Уверенность модели (0-1)
    'probabilities': {
        'team1': 0.61,               # Вероятность победы Team 1
        'team2': 0.39                # Вероятность победы Team 2
    },
    'ratings': {
        'team1': 8,                  # Рейтинг HLTV Team 1
        'team2': 11,                 # Рейтинг HLTV Team 2
        'delta': 3                   # Разница рейтингов
    },
    'h2h': {
        'shrunk': -2,                # Личные встречи (отрицательное = Team 1 выигрывает чаще)
        'count': 6                   # Количество личных встреч
    }
}
```

---

## Примеры использования

### Пример 1: Быстрое предсказание

```python
from simple_predict import SimplePredictor

predictor = SimplePredictor()

result = predictor.predict_by_names(
    team1_name="G2",
    team2_name="Vitality",
    is_lan=1,
    map_name="Dust2"
)

# Результат выводится автоматически
# >>> Победитель: G2
#     Уверенность: 65.3%

predictor.close()
```

### Пример 2: Программное использование

```python
from simple_predict import SimplePredictor

predictor = SimplePredictor()

# Получить результат без вывода
result = predictor.predict_by_names(
    team1_name="Spirit",
    team2_name="Mouz",
    is_lan=0
)

if result and 'error' not in result:
    winner = result['winner']
    confidence = result['confidence']

    if confidence > 0.65:
        print(f"Уверенное предсказание: {winner} ({confidence:.1%})")
    else:
        print(f"Неуверенное предсказание: {winner} ({confidence:.1%})")
        print("Не рекомендуется для ставок")

predictor.close()
```

### Пример 3: BO3 с анализом

```python
from simple_predict import SimplePredictor

predictor = SimplePredictor()

results = predictor.predict_bo3(
    team1_name="Natus Vincere",
    team2_name="FaZe",
    maps=["Dust2", "Mirage", "Inferno"],
    is_lan=1
)

# Проанализировать результаты
for i, result in enumerate(results, 1):
    print(f"Карта {i}: {result['winner']} ({result['confidence']:.1%})")

predictor.close()
```

### Пример 4: Поиск команды

```python
from simple_predict import SimplePredictor

predictor = SimplePredictor()

# Найти команду по части названия
teams = predictor.find_team("Natus")
print(teams)
# Выведет:
#    team_id      team_name
# 0        4  Natus Vincere

predictor.close()
```

---

## Частые вопросы

### Команда не найдена?

Попробуй:
1. Написать часть названия: "Natus" вместо "Natus Vincere"
2. Проверить написание
3. Использовать `find_team()` для поиска

### Низкая уверенность (<60%)?

Это нормально для близких матчей. Не делай ставки на такие матчи.

### Нужно обновить рейтинги?

Рейтинги берутся из БД. Обновляй БД еженедельно (парсинг HLTV).

### Модель ошибается?

Точность модели 62.27% - это значит ~38% матчей будут предсказаны неправильно. Это нормально для CS2.

---

## Альтернативный способ (через ID)

Если знаешь ID команд, можешь использовать `predict_match.py` напрямую:

```python
from predict_match import MatchPredictor

predictor = MatchPredictor()

result = predictor.predict(
    team1_id=4,      # ID команды
    team2_id=2,      # ID команды
    is_lan=1,
    map_name="Mirage"
)

print(f"Победитель: {result['winner']}")
print(f"Уверенность: {result['confidence']:.1%}")

predictor.close()
```

---

## Итого

**Для большинства случаев используй `simple_predict.py`:**

```python
from simple_predict import SimplePredictor

predictor = SimplePredictor()

# Один матч
result = predictor.predict_by_names("NaVi", "FaZe", is_lan=1, map_name="Mirage")

# BO3
results = predictor.predict_bo3("NaVi", "FaZe", ["Dust2", "Mirage", "Inferno"], is_lan=1)

predictor.close()
```

Готово!
