# Выводы по exp008

`exp008` развивает успешный `exp006`: постановка остается строго candidate-level.

Обучающая единица:

`(exam_row_id, assigned_factor) -> factor_is_contraindicated`

Главное изменение: вместо одного глобального threshold для всех факторов используются отдельные thresholds по частым факторам.

Почему это важно:

- base rate у факторов разный;
- `6.1`, `14`, `15`, `18.1`, `6.2` не должны иметь один и тот же decision boundary;
- метрика считает множество причин, поэтому неверный порог по одному частому factor портит Jaccard на многих строках.

Validation:

- `Autonomous Score`: `0.47647`
- `F1(any)`: `0.397906`
- `Jaccard`: `0.555034`
- validation positive rate: `10.79%`

Test submission:

- positive rows: `692 / 6215`
- positive rate: `11.13%`
- subset violations: `0`

Сравнение:

- exp006: local `0.44209`, test positive `10.07%`
- exp008: local `0.47647`, test positive `11.13%`

Вывод: это лучший candidate-level вариант на данный момент. Он сохраняет правильную постановку задачи и заметно улучшает local score за счет factor-specific thresholds.
