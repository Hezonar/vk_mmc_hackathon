# Experiments

Каждый эксперимент хранится в отдельной папке:

- `params.yml` - гиперпараметры и пути.
- `info.yml` - короткое описание гипотезы и результата.
- `train.py` - обучение, подбор порогов на validation и генерация `submission.csv`.
- `inference.py` - повторный inference с тем же кодом эксперимента.
- `metrics.json` - метрики на split 70/30.
- `validation_predictions.csv` - предсказания на validation.
- `submission.csv` - файл для загрузки.

Основная локальная метрика:

`Autonomous Score = alpha * F1(any contraindication) + (1 - alpha) * mean Jaccard(factors on true-positive rows)`

Если `alpha` в соревновании не раскрыт, в экспериментах используем `alpha=0.5` как нейтральную настройку.
