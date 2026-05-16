# Выводы по exp005

После leaderboard стало понятно, что `exp001` был лучшим не из-за максимального local score, а из-за правильного профиля предсказаний:

- `exp001`: leaderboard `public=0.41902`, `private=0.41650`
- test positive rate: `8.48%`

`exp004` локально был лучше, но слишком сильно изменил профиль:

- test positive rate: `18.83%`
- leaderboard стал хуже: `public=0.41262`, `private=0.38392`

`exp005` возвращается к стилю `exp001`, но усиливает векторизацию:

- word TF-IDF `(1,2)`
- char_wb TF-IDF `(3,5)` для опечаток, сокращений и вариантов медицинских формулировок
- без retrieval
- без group split
- жесткий штраф за positive rate вне диапазона около `8-10%`

Validation:

- `Autonomous Score`: `0.473581`
- `F1(any)`: `0.415686`
- `Jaccard`: `0.531476`
- validation positive rate: `9.10%`

Test submission:

- positive rows: `620 / 6215`
- positive rate: `9.98%`

Вывод: это наиболее разумный следующий submission после провала `exp004`, потому что он улучшает локальные метрики относительно `exp001`, но сохраняет близкий к лучшему leaderboard positive-rate профиль.
