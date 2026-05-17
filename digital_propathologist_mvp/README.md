# Цифровой профпатолог

Рабочее место врача-профпатолога для проверки профосмотров, оценки вредных факторов ML-моделью и формирования отчета.

## Возможности

- загрузка CSV с профосмотрами;
- поиск пациента по `patient_id`;
- очередь профосмотров с дневным счетчиком проверок;
- выбор конкретного `exam_row_id`;
- отображение факторов вредности, заключений специалистов, исследований, рекомендаций и контрольных подсказок;
- ML-оценка `Годен / Не годен / Недостаточно данных`;
- подсветка медицинских оснований при наведении на фактор;
- сохранение завершенных проверок в `data/review_state.json`;
- выгрузка CSV-ответа и текстового отчета.

## Запуск

```bash
cd digital_propathologist_mvp
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Данные

Ожидаемые колонки:

- `exam_row_id`
- `medical_exam_id`
- `patient_id`
- `consultation_date`
- `assigned_harmful_factors`
- `specialist_conclusions`
- `contraindicated_factors`
- `has_contraindications`

## ML-модель

Интерфейс использует артефакт:

```text
experiments/006_candidate_factor_binary/model.pkl
```

Если файл отсутствует, обучите модель командой из корня репозитория:

```bash
python experiments/006_candidate_factor_binary/train.py
```
