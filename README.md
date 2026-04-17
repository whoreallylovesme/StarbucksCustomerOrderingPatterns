# Starbucks Customer Ordering Patterns -- MLOps Pipeline

MLOps MVP для потоковой обработки данных на датасете
[Starbucks Customer Ordering Patterns](https://www.kaggle.com/datasets/likithagedipudi/starbucks-customer-ordering-patterns).

Задача: регрессия -- предсказание среднего чека (`total_spend`, в долларах).
Модель: CatBoostRegressor с инкрементальным обучением (init_model).

---

## Установка

```bash
pip install -r requirements.txt
```

Путь к датасету задается в `configs/data.yaml` (поле `data.source_file`).
По умолчанию: `artifacts/data/starbucks_customer_ordering_patterns.csv`.
Для автоматической загрузки с Kaggle установите `data.kaggle.enabled: true`
в том же файле (требуется `~/.kaggle/kaggle.json`).

---

## Использование

```bash
# Запустить пайплайн на следующем батче данных
python run.py -mode "update"

# Применить production модель к новым данным
# Возвращает путь к CSV с добавленной колонкой "predict"
python run.py -mode "inference" -file "./path/to/data.csv"

# Сгенерировать Markdown-отчет о работе системы
python run.py -mode "summary"
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
runtime/                      пакет с логикой пайплайна
    ingestion.py              батчевый стриминг, raw store, метапараметры
    analysis.py               DQ, EDA, statsmodels анализ, data drift
    preparation.py            feature engineering, импутация
    training.py               инкрементальное обучение CatBoost
    validation.py             hold-out оценка, реестр моделей, model drift
    serving.py                сериализация, inference, мониторинг
    lib/
        io.py                 общие утилиты JSON/pickle
artifacts/                    все генерируемые файлы
    data/
        starbucks_customer_ordering_patterns.csv
        raw/                  batch_000.csv ... batch_009.csv
        meta/                 state.json, meta_batch_NNN.json
        quality/              dq_batch_NNN.json, eda_batch_NNN.json
        predictions/          predictions_YYYYMMDD_HHMMSS.csv
    models/
        registry.json         история версий моделей с метриками
        catboost_incremental.pkl  текущий инкрементальный чекпоинт
        production/           latest.pkl (модель + препарер)
    logs/
        pipeline.log
        monitoring.json       история вызовов inference (latency, память)
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
нарезает на батчи, считает метапараметры (распределение таргета, % пропусков,
диапазон дат) и сохраняет `state.json` -- указатель на следующий батч.

При каждом следующем `update`: возвращает очередной необработанный батч.

#### 2. Анализ данных (analysis.py)

Data Quality -- считает % пропусков по колонкам, % дублей, выбросы по z-score.
Сохраняет `dq_batch_NNN.json`. Колонки с пропусками выше порога удаляются.

Auto EDA -- описательные статистики числовых колонок (mean/median/std/skew),
топ-5 значений категориальных, корреляции Пирсона с таргетом.
Сохраняет `eda_batch_NNN.json`.

Statsmodels EDA -- OLS регрессия для получения p-value каждой числовой фичи.
Незначимые фичи (p > 0.05) исключаются из обучения. VIF проверяет
мультиколлинеарность, тест Харке-Бера -- нормальность распределений.
Набор фичей фиксируется на нулевом батче и не меняется -- это необходимо
для совместимости инкрементальной модели между батчами.

Data Drift -- KS-тест между текущим и предыдущим батчем.
При значимом сдвиге распределения выдается предупреждение в лог.

#### 3. Подготовка данных (preparation.py)

- Feature engineering: `order_time` -> `hour`; `order_date` -> `month`, `year`
- Удаление ID-колонок, временных колонок, незначимых фичей
- Заполнение пропусков: медианой для числовых, "Unknown" для категориальных
- Возвращает DataFrame -- CatBoost работает с категориальными нативно, OHE не нужен

#### 4. Обучение (training.py)

Инкрементальное обучение через init_model:

```
батч 0  ->  обучение с нуля          ->  100 деревьев
батч 1  ->  fit(init_model=пред.)    ->  200 деревьев
батч N  ->  fit(init_model=пред.)    ->  (N+1) x 100 деревьев
```

Каждый батч добавляет 100 деревьев поверх предыдущей модели
без обучения с нуля. Чекпоинт сохраняется в
`artifacts/models/catboost_incremental.pkl`.

#### 5. Валидация (validation.py)

- Оценка на 20% hold-out от накопленных данных: RMSE, MAE, R2
- Каждая версия модели сохраняется в `artifacts/models/` -> `registry.json`
- Промоушн: если RMSE улучшилась на >= 0.01 относительно лучшей предыдущей
- Model drift: если последняя RMSE хуже исторического минимума на > 0.1 -> warning

#### 6. Обслуживание (serving.py)

- Сериализует production модель как pickle-бандл: модель + препарер -> `latest.pkl`
- Inference: загружает бандл, прогоняет через препарер, вызывает `model.predict()`
- Измеряет latency (мс) и пик памяти (КБ) через `tracemalloc`
- Логирует каждый вызов в `artifacts/logs/monitoring.json`

### Поток данных при update

```
ingestion   ->  следующий батч (10k строк)
    |
analysis    ->  проверка качества + EDA + data drift
    |
preparation ->  feature engineering + импутация
    |
training    ->  +100 деревьев поверх предыдущей модели
    |
validation  ->  RMSE/MAE/R2 на hold-out -> промоушн если лучше
    |
serving     ->  artifacts/models/production/latest.pkl
```

---

## Результаты

После 10 батчей (100 000 строк, 1000 деревьев итого):

| Метрика | Лучший результат (батч 2) |
|---------|--------------------------|
| RMSE    | 1.38                     |
| MAE     | 0.97                     |
| R2      | 0.936                    |

Средний чек в датасете: ~$14.87. Ошибка ~$1.38 -- около 9% от среднего.
