# Match Outcome Predictor

Система машинного обучения для предсказания результатов киберспортивных матчей на основе рейтинга команд, истории личных встреч и контекста матча.

## 🎯 Основные результаты

- **Точность модели:** 62.27%
- **Алгоритм:** Logistic Regression (C=0.5)
- **Улучшение над baseline:** +2.06%
- **Baseline (простое правило):** 60.21%

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Обучение модели

```bash
python main.py
```

Вывод:
```
Accuracy: 0.6227 (62.27%)
Model saved to: model.pkl
```

### Предсказание для одного матча

```python
from predict import predict_match

prediction, confidence = predict_match(
    delta_rating=-22,      # Team2_rating - Team1_rating
    map_id=4,
    delta_map_wr_windowed=0.0,
    map_missing_flag=0,
    LAN_flag=1,
    h2h_shrunk=0,
    h2h_count=0
)

print(f"Победитель: {'Команда 1' if prediction == -1 else 'Команда 2'}")
print(f"Уверенность: {confidence:.1%}")
```

### Пакетное предсказание

```python
from predict import predict_from_csv

results = predict_from_csv('new_matches.csv')
print(results[['prediction', 'confidence']])
```

## 📊 Структура проекта

```
predictor/
├── main.py                      # Основной скрипт обучения
├── predict.py                   # Интерфейс для предсказаний
├── model.pkl                    # Обученная модель
├── requirements.txt             # Зависимости
├── README.md                    # Этот файл
│
├── train/                       # Обучающие данные
│   ├── x_train.csv             # Признаки (3096 матчей)
│   └── y_train.csv             # Результаты (-1 или +1)
│
├── test/                        # Тестовые данные
│   ├── x_test.csv              # Признаки (774 матча)
│   └── y_test.csv              # Результаты
│
├── analysis/                    # Аналитические скрипты
│   ├── baseline_comparison.py   # Сравнение с baseline
│   ├── optimized_model.py       # Результаты ablation study
│   ├── hyperparameter_tuning.py # Подбор гиперпараметров SVM
│   ├── model_comparison.py      # Сравнение алгоритмов
│   ├── feature_engineering.py   # Тестирование преобразований
│   ├── error_analysis.py        # Анализ ошибок модели
│   ├── best_model_final.py      # Финальная модель с метриками
│   ├── final_model.py           # Альтернативная финальная модель
│   └── verify_baseline.py       # Проверка baseline
│
└── docs/                        # Документация
    ├── FINAL_REPORT.md          # Полный отчет по анализу
    ├── ANALYSIS_SUMMARY.md      # Краткие выводы
    ├── CORRECTED_SUMMARY.md     # Исправленный анализ
    ├── CRITICAL_FINDING.md      # Критическое открытие
    └── PROJECT_SUMMARY.md       # Итоговая сводка
```

## 🔍 Признаки (Features)

### Исходные признаки (7 штук)

| Признак | Описание | Использование |
|---------|----------|---------------|
| `delta_rating` | Разница рейтингов (Team2 - Team1) | ✅ Используется |
| `map_id` | Идентификатор карты | ❌ Удален (ухудшает на 0.52%) |
| `delta_map_wr_windowed` | Разница винрейта на карте | ❌ Удален (нет эффекта) |
| `map_missing_flag` | Отсутствуют данные по карте | ✅ Используется |
| `LAN_flag` | Матч на LAN | ✅ Используется |
| `h2h_shrunk` | История личных встреч (сжатая) | ✅ Используется |
| `h2h_count` | Количество личных встреч | ✅ Используется |

## 📈 Результаты экспериментов

### 1. Baseline Comparison

| Модель | Точность | Примечание |
|--------|----------|------------|
| **Простое правило ** | **60.21%** | delta > 0 => -1 |
| SVM (все признаки) | 60.47% | +0.26% |
| SVM (оптимизированные) | 61.63% | +1.42% |
| **Logistic Regression** | **62.27%** | **+2.06%** ✨ |

### 2. Сравнение алгоритмов

Протестировано 4 алгоритма с различными гиперпараметрами:

- **Logistic Regression:** 62.27% (лучший)
- Random Forest: 62.14%
- Gradient Boosting: 62.02%
- SVM: 61.63%

### 3. Feature Engineering

Протестированы различные преобразования признаков:

- Polynomial features (degree=2): **Хуже** (61.63%)
- Interaction terms (delta × h2h): **Хуже** (62.14%)
- Log transform: **Хуже** (60.59%)

**Вывод:** Оригинальные признаки оптимальны, сложные преобразования добавляют шум.

### 4. Важность признаков

```
delta_rating:      -0.3978  ← Самый сильный предиктор
h2h_shrunk:         0.2986  ← Второй по важности
map_missing_flag:   0.0223
h2h_count:          0.0208
LAN_flag:           0.0195
```

## 🔮 Будущие улучшения

Для значительного улучшения точности (>62%) необходимы:

### 1. Новые данные
- Недавняя форма (последние 5-10 игр)
- Данные по игрокам (если доступны)
- Контекст турнира (важность матча)
- Индикаторы усталости (игры за последние 24ч)

### 2. Улучшения модели
- Балансировка классов (class weights)
- Ансамбли (ensemble methods)
- Калибровка вероятностей
- Оптимизация порога решения

### 3. Валидация
- Временная валидация (не случайное разделение)
- Кросс-валидация
- A/B тестирование в продакшене

## 📚 Документация

Все документы находятся в папке `docs/`:

- **docs/FINAL_REPORT.md** - Полный отчет со всеми экспериментами
- **docs/ANALYSIS_SUMMARY.md** - Ключевые инсайты и рекомендации
- **docs/CORRECTED_SUMMARY.md** - Исправленный анализ baseline
- **docs/CRITICAL_FINDING.md** - Критическое открытие об обратной зависимости
- **docs/PROJECT_SUMMARY.md** - Итоговая сводка проекта

## 🔬 Воспроизведение экспериментов

Все аналитические скрипты находятся в папке `analysis/`. Запускайте их из этой папки:

```bash
cd analysis

# Сравнение с baseline
python baseline_comparison.py

# Проверка baseline
python verify_baseline.py

# Подбор гиперпараметров
python hyperparameter_tuning.py

# Сравнение моделей
python model_comparison.py

# Feature engineering
python feature_engineering.py

# Анализ ошибок
python error_analysis.py

# Финальная модель
python best_model_final.py
```

## 🛠️ Технические детали

### Требования

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

### Confusion Matrix

```
                Predicted
                -1    +1
Actual  -1     193   185
        +1     107   289
```

### Метрики

- **Accuracy:** 62.27%
- **Precision (Team 1):** 64%
- **Precision (Team 2):** 61%
- **Recall (Team 1):** 51%
- **Recall (Team 2):** 73%
## 📝 Лицензия

Проект создан в образовательных и исследовательских целях.