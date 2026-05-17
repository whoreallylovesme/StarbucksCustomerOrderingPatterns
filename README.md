# Starbucks Customer Ordering Patterns — MLOps Pipeline

MLOps MVP для потоковой обработки данных на датасете
[Starbucks Customer Ordering Patterns](https://www.kaggle.com/datasets/likithagedipudi/starbucks-customer-ordering-patterns).

Задача: регрессия — предсказание среднего чека (`total_spend`, в долларах).
Модель: CatBoostRegressor с инкрементальным обучением (init_model).

---

## Установка

```bash
pip install -r requirements.txt
```

Путь к датасету задаётся в `configs/data.yaml` (поле `data.source_file`).
По умолчанию: `artifacts/data/starbucks_customer_ordering_patterns.csv`.

Для автоматической загрузки с Kaggle установите `data.kaggle.enabled: true`
в `configs/data.yaml` и передайте credentials через переменные окружения:

```bash
KAGGLE_USERNAME=<логин> KAGGLE_KEY=<токен> python run.py -mode "update"
```

Токен берётся на странице https://www.kaggle.com/settings -> API -> Create New Token.

---

## Использование

```bash
# Запустить пайплайн на следующем батче данных
python run.py -mode "update"

# Применить production модель к новым данным
python run.py -mode "inference" -file "./path/to/data.csv"

# Сгенерировать отчёт о работе системы
python run.py -mode "summary"
```

---

## CI/CD (GitHub Actions)

### Подход

Реализован сценарий **CRON-инкрементального обучения**. Система спроектирована
под инкремент: CatBoost дообучается через `init_model`, добавляя каждый запуск
`iterations_per_batch` деревьев поверх предыдущей модели. CRON-расписание
(ежедневно в 06:00 UTC) эмулирует появление нового батча потоковых данных.
Модель и состояние пайплайна сохраняются между запусками через кэш GitHub Actions.

### Архитектура

```
push / pull_request / CRON (ежедневно) / ручной запуск
              |
              v
         [test]  — pytest tests/
              |
              v
         [train] — восстановление кэша модели с предыдущего запуска
                   -> загрузка датасета с Kaggle (если нет в кэше)
                   -> python run.py -mode update
                   -> python run.py -mode summary
                   -> сохранение кэша и артефактов
```

### Настройка

Добавьте секреты в `Settings -> Secrets and variables -> Actions`:

| Secret | Значение |
|--------|----------|
| `KAGGLE_USERNAME` | логин на kaggle.com |
| `KAGGLE_KEY` | токен из kaggle.com/settings -> API -> Create New Token |

Ручной запуск: `Actions -> MLOps CI/CD -> Run workflow`

### Артефакты

| Артефакт | Содержимое | Хранится |
|----------|-----------|----------|
| `training-logs-N` | `pipeline.log`, `monitoring.json` | 30 дней |
| `production-model-N` | `latest.pkl` (модель + препарер) | 30 дней |
| `summary-report-N` | отчёт о метриках | 30 дней |

### Тесты

```bash
pytest tests/ -v
```

---

## Структура проекта

```
run.py                        точка входа
requirements.txt
configs/
    data.yaml                 источник данных, размер батча, пути
    analysis.yaml             пороги качества, настройки подготовки
    training.yaml             параметры CatBoost, валидация, промоушн
    serving.yaml              пути к production модели, логам, отчетам
runtime/
    ingestion.py              батчевый стриминг, raw store, метапараметры
    analysis.py               DQ, EDA, statsmodels анализ, data drift
    preparation.py            feature engineering, импутация
    training.py               инкрементальное обучение CatBoost
    validation.py             hold-out оценка, реестр моделей, model drift
    serving.py                сериализация, inference, мониторинг
    lib/
        io.py                 общие утилиты JSON/pickle
tests/
    conftest.py               фикстуры pytest
    test_config.py            валидность конфигов
    test_preparer.py          подготовка данных
    test_model.py             обучение и метрики
artifacts/
    data/
        raw/                  batch_000.csv ... batch_009.csv
        meta/                 state.json, meta_batch_NNN.json
        quality/              dq_batch_NNN.json, eda_batch_NNN.json
        predictions/          predictions_YYYYMMDD_HHMMSS.csv
    models/
        registry.json
        catboost_incremental.pkl
        production/           latest.pkl (модель + препарер)
    logs/
        pipeline.log
        monitoring.json
    reports/                  summary_YYYYMMDD_HHMMSS.md
```

---

## Как работает система

### Датасет

100 000 строк x 20 колонок. Временная переменная: `order_date`.
Данные сортируются по дате и нарезаются на 10 батчей по 10 000 строк.
При инициализации в 5 колонок инжектируется 5% пропусков
(`fulfillment_time_min`, `cart_size`, `num_customizations`,
`order_channel`, `drink_category`).

### Этапы пайплайна

#### 1. Сбор данных (ingestion.py)

При первом запуске: читает CSV, сортирует по `order_date`, инжектирует пропуски,
нарезает на батчи, считает метапараметры и сохраняет `state.json`.

При каждом следующем `update`: возвращает очередной необработанный батч.

#### 2. Анализ данных (analysis.py)

Data Quality — считает % пропусков по колонкам, % дублей, выбросы по z-score.
Сохраняет `dq_batch_NNN.json`.

Auto EDA — описательные статистики, топ-5 категориальных значений, корреляции с таргетом.

Statsmodels EDA — OLS регрессия для получения p-value фичей. Незначимые (p > 0.05)
исключаются. Набор фичей фиксируется на нулевом батче.

Data Drift — KS-тест между текущим и предыдущим батчем.

#### 3. Подготовка данных (preparation.py)

- Feature engineering: `order_time` -> `hour`; `order_date` -> `month`, `year`
- Удаление ID-колонок и незначимых фичей
- Заполнение пропусков: медианой для числовых, "Unknown" для категориальных

#### 4. Обучение (training.py)

```
батч 0  ->  обучение с нуля        ->  100 деревьев
батч 1  ->  fit(init_model=пред.)  ->  200 деревьев
батч N  ->  fit(init_model=пред.)  ->  (N+1) x 100 деревьев
```

#### 5. Валидация (validation.py)

- Оценка на 20% hold-out: RMSE, MAE, R²
- Промоушн: если RMSE улучшилась на ≥ 0.01 относительно лучшей предыдущей
- Model drift: если RMSE хуже исторического минимума на > 0.1 -> warning

#### 6. Обслуживание (serving.py)

- Сериализует production модель: модель + препарер -> `latest.pkl`
- Inference: загружает бандл, прогоняет через препарер, вызывает `predict()`
- Измеряет latency (мс) и пик памяти (КБ) через `tracemalloc`

---

## Результаты

После 10 батчей (100 000 строк, 1000 деревьев итого):

| Батч | RMSE   | MAE    | R²     | В production |
|------|--------|--------|--------|--------------|
| 0    | 1.4485 | 0.9833 | 0.9292 | да           |
| 1    | 1.4162 | 0.9775 | 0.9333 | да           |
| 2    | 1.3765 | 0.9681 | 0.9358 | да (лучшая)  |
| 3    | 1.4358 | 0.9833 | 0.9313 | нет          |
| 4    | 1.4429 | 0.9892 | 0.9318 | нет          |
| 5    | 1.4232 | 0.9838 | 0.9331 | нет          |
| 6    | 1.4394 | 0.9890 | 0.9309 | нет          |
| 7    | 1.4015 | 0.9747 | 0.9357 | нет          |
| 8    | 1.4166 | 0.9763 | 0.9327 | нет          |
| 9    | 1.4253 | 0.9799 | 0.9333 | нет          |

Лучшая модель — батч 2: RMSE=1.38, MAE=0.97, R²=0.936.
Средний чек в датасете: ~$14.87. Ошибка ~$1.38 — около 9% от среднего.
