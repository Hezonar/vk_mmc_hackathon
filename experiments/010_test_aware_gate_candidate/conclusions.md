# Выводы по exp010

`exp010` использует test без таргетов:

- TF-IDF vocabulary fit на `train + test`;
- labels используются только из train;
- threshold подбирается по validation, но с constraint на test positive rate;
- архитектура: `row-level gate -> candidate-level factor classifier`.

Лучшие параметры:

- `row_threshold`: `0.4`
- `factor_threshold`: `0.45`
- `max_factors`: `2`

Validation:

- `Autonomous Score`: `0.447759`
- `F1(any)`: `0.398157`
- `Jaccard`: `0.497361`
- validation positive rate: `9.97%`

Test submission:

- positive rows: `675 / 6215`
- positive rate: `10.86%`
- subset violations: `0`

Сравнение:

- exp006 local: `0.44209`, test positive `10.07%`, public `0.44593`
- exp010 local: `0.447759`, test positive `10.86%`
- exp008 local: `0.47647`, test positive `11.13%`, не проверен на leaderboard

Вывод: exp010 немного улучшает exp006 локально и использует test-distribution легально. Это разумный следующий кандидат, хотя exp008 локально сильнее.
