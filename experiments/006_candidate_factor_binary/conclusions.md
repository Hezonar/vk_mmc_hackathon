# Выводы по exp006

`exp006` меняет постановку задачи на точную:

> для каждого `assigned factor` предсказать, противопоказан именно он или нет.

То есть обучающая единица теперь не строка осмотра, а пара:

`(exam_row_id, assigned_factor) -> 0/1`

Потом положительные candidates агрегируются обратно в CSV `exam_row_id,factors`.

Важные свойства:

- output всегда subset от `assigned_harmful_factors`;
- нет предсказаний кодов, которых не было во входе;
- нет искусственного one-vs-rest списка всех факторов;
- модель учится на `77 956` candidate-строках train split;
- candidate positive rate всего `1.44%`, задача сильно несбалансирована.

Validation:

- `Autonomous Score`: `0.44209`
- `F1(any)`: `0.392558`
- `Jaccard`: `0.491621`
- validation positive row rate: `9.84%`

Test submission:

- positive rows: `626 / 6215`
- positive rate: `10.07%`
- subset violations: `0`

Сравнение:

- exp001: leaderboard best so far, local `0.417749`, test positive `8.48%`
- exp005: local `0.473581`, test positive `9.98%`
- exp006: local `0.44209`, test positive `10.07%`

Локально exp006 хуже exp005, но он ближе к правильной математической постановке задачи. Его стоит проверить на leaderboard: если метрика/скрытый датасет действительно лучше соответствует per-candidate классификации, он может оказаться устойчивее row-level one-vs-rest подходов.
