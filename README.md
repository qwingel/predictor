# predictor_runtime

Автоматический пайплайн предсказаний CS2: парсинг HLTV → прогноз → публикация в Telegram.

## Описание

Runtime-часть проекта — делает предсказания на основе уже обученной модели. Парсит расписание матчей с hltv.org, генерирует прогнозы для всех карт, парсит коэффициенты Winline, формирует рекомендации по ставкам и публикует отчёт в Telegram.

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Файлы данных

| Файл | Описание |
|------|----------|
| `data/cs2_data.db` | SQLite база с историей матчей (нужна для расчёта признаков) |
| `data/top_teams.txt` | Рейтинги команд |
| `data/predictions_log.json` | Лог всех прогнозов (для последующей проверки точности) |
| `data/accuracy_tracker.json` | Кумулятивная статистика точности |
| `models/model_lgbm_final.pkl` | Обученная модель |

## Использование

### Автоматический пайплайн

```bash
python auto_pipeline.py
```

Выполняет:
1. **Парсинг HLTV** — находит матчи топ-30 команд на сегодня через Selenium
2. **Предсказания** — для каждого матча прогнозирует исход всех 7 карт + общий победитель
3. **Winline odds** — парсит коэффициенты букмекера
4. **Bet recommendations** — рассчитывает EV и рекомендацию (СТАВИТЬ / РИСК / НЕ СТАВИТЬ)
5. **Telegram** — публикует форматированный отчёт
6. **Лог** — сохраняет прогнозы в `data/predictions_log.json`

### Проверка точности (на следующий день)

```bash
python results_report.py
```

Сравнивает вчерашние прогнозы с реальными результатами матчей, парсит счёты с HLTV и публикует отчёт об точности.

### Ручное предсказание

```python
from predict import load_model, predict_match, predict_for_all_maps

model = load_model('models/model_lgbm_final.pkl')

# Все карты матча
result = predict_for_all_maps(model, 'Vitality', 'Furia', is_lan=1)
for map_name, data in result.items():
    winner, _, p1, p2 = data
    print(f"{map_name}: {winner} ({max(p1, p2):.0%})")

# Одна карта
result = predict_match(model, 'NAVI', 'FaZe', 'Mirage', is_lan=0)
print(f"P(team1)={result['prob_team1']:.2%}, P(team2)={result['prob_team2']:.2%}")
```

### Рекомендации по ставкам

```python
from bet_recommendation import evaluate_bet

bet = evaluate_bet(model_prob=0.65, bookmaker_odds=1.83)
print(bet['recommendation'])     # СТАВИТЬ / РИСК / НЕ СТАВИТЬ
print(bet['ev'])                 # математическое ожидание
print(bet['bet_size_percent'])   # размер ставки по Келли
```

## Структура

```
predictor_runtime/
├── auto_pipeline.py              # Главный пайплайн: HLTV → прогноз → Telegram
├── predict.py                    # API предсказаний (predict_match, predict_for_all_maps)
├── parse_winline.py              # Парсинг коэффициентов Winline
├── bet_recommendation.py         # Расчёт EV, Kelly criterion, рекомендации
├── results_report.py             # Сравнение прогнозов с реальными результатами
├── telegram_publisher.py         # Публикация в Telegram (через API-прокси)
├── .env                          # API_URL_TO_POST, API_SECRET_KEY
├── ml/
│   ├── calibration.py            # Temperature scaling
│   ├── config.py                 # Константы команд (TOP_30, алиасы)
│   ├── team_matcher.py           # Сопоставление имён команд
│   └── utils.py                  # Утилиты (нормализация, веса, рейтинги)
├── models/
│   └── model_lgbm_final.pkl      # Готовая модель
└── data/
    ├── cs2_data.db               # База матчей
    ├── top_teams.txt             # Рейтинги команд
    ├── predictions_log.json      # Лог прогнозов
    └── accuracy_tracker.json     # Трекер точности
```

## Telegram

Для публикации требуется `.env` с переменными:

```
API_URL_TO_POST=https://your-api-endpoint/post
API_SECRET_KEY=your-secret-key
```

## Зависимости

```
pandas>=2.0.0
numpy>=1.24.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
joblib>=1.3.0
tqdm>=4.65.0
undetected-chromedriver
selenium
requests
python-dotenv
```

## Особенности

1. **Нормализация имён команд**: Все названия автоматически приводятся к lowercase с применением маппинга известных вариаций (NAVI -> natus vincere, Furia -> furia, и т.д.)

2. **Симметричность модели**: При перестановке команд местами предсказание инвертируется: `P(team1, team2) = 1 - P(team2, team1)`. Достигается двойным проходом с инвертированными признаками.

3. **Temperature scaling**: Калибровка вероятностей после симметризации для улучшения качества предсказаний.

4. **Kelly criterion**: Оптимальный размер ставки рассчитывается по дробному Келли (25%) с ограничением 1-5% от банка.
