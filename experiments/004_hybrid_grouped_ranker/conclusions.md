# Выводы по exp004

`exp004` объединяет самые полезные идеи из предыдущих запусков:

- `GroupShuffleSplit` по `patient_id`, чтобы validation был честнее.
- Binary-решение оставлено за TF-IDF + LogisticRegression.
- Причины ранжируются гибридно: `0.55 * factor_model + 0.35 * retrieval + 0.10 * prior`.
- Порог подбирается по среднему score для `alpha=[0.3, 0.5, 0.7]` со штрафом за слишком высокий positive rate.

Лучшие параметры:

- `any_threshold`: `0.4`
- `factor_threshold`: `0.45`
- `max_factors`: `3`
- `fallback_top_factor`: `false`

Validation:

- `Autonomous Score`, alpha `0.5`: `0.492897`
- `F1(any)`: `0.307256`
- `Jaccard`: `0.678538`
- validation positive rate: `19.13%`

Test submission:

- positive rows: `1170 / 6215`
- positive rate: `18.83%`

Сравнение:

- exp001: `0.417749`, test positive `8.48%`
- exp002: `0.483131`, test positive `28.61%`
- exp003: `0.427239`, test positive `88.99%`
- exp004: `0.492897`, test positive `18.83%`

Вывод: exp004 выглядит лучше exp002 не только по локальному score, но и по риску. Он сохраняет высокий recall, улучшает F1(any) и не уходит в чрезмерно агрессивные positive-предсказания.
