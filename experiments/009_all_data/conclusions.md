# Выводы по exp009_all_data

`exp009_all_data` - production-версия `exp006`.

Что сделано:

- модель обучена на всех `data/train2.csv`;
- обучающая единица осталась правильной: `(exam_row_id, assigned_factor) -> 0/1`;
- обучено `111 029` candidate-строк;
- артефакт сохранен в `src/app/profpath/model_artifacts/exp009_all_data.joblib`;
- приложение теперь использует этот артефакт вместо старой ML-заглушки;
- test inference выполнен, submission сохранен.

Submission:

- rows: `6215`
- positive rows: `626`
- positive rate: `10.07%`
- threshold: `0.53`

Интеграция:

- добавлен `app.profpath.ml_model.predict_with_candidate_model`;
- `predict_row_stub(..., use_ml_stub=True)` сначала использует реальную модель, а если артефакта нет, безопасно откатывается к старой заглушке;
- контракт результата сохранен: `risk_score`, `has_contraindications_pred`, `contraindicated_factors_pred`, `risk_level`, `model_explanation`, `model_version`.
