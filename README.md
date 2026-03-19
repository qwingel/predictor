# CS2 Match Predictor

Система предсказания исходов матчей CS2 с точностью 62.27%.

## Файлы

### Основные скрипты
- **simple_predict.py** - главный скрипт для предсказаний (используй этот)
- **predict_match.py** - базовый скрипт (если нужны ID команд)
- **monitor_model.py** - мониторинг качества модели
- **retrain_model.py** - переобучение модели (раз в 3-6 месяцев)

### Данные
- **best_model_logistic.pkl** - обученная модель (62.27% точности)
- **hltv_data.db** - база данных (42k матчей, рейтинги команд)

### Документация
- **КАК_ИСПОЛЬЗОВАТЬ.md** - полная инструкция с примерами

## Быстрый старт

```python
from simple_predict import SimplePredictor

predictor = SimplePredictor()

# Предсказать один матч
result = predictor.predict_by_names(
    team1_name="Natus Vincere",
    team2_name="FaZe",
    is_lan=1,
    map_name="Mirage"
)

# BO3 (три карты)
results = predictor.predict_bo3(
    team1_name="Natus Vincere",
    team2_name="FaZe",
    maps=["Dust2", "Mirage", "Inferno"],
    is_lan=1
)

predictor.close()
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Обслуживание

- **Еженедельно:** обновлять данные (парсинг HLTV)
- **Ежемесячно:** проверять точность (`monitor_model.py`)
- **Раз в 3-6 месяцев:** переобучать модель (`retrain_model.py`)

## Подробности

Читай **КАК_ИСПОЛЬЗОВАТЬ.md** для полной инструкции.
