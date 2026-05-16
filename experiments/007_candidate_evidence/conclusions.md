# Выводы по exp007

Идея: добавить к candidate-level модели историческое evidence из похожих позитивных случаев того же factor.

Результат оказался плохим:

- local `Autonomous Score`: `0.17781`
- `F1(any)`: `0.151515`
- `Jaccard`: `0.204106`
- test positive rate: `8.54%`

Причина: same-factor positive retrieval слишком разреженный. Для многих факторов мало позитивных примеров, а similarity-сигнал плохо калибруется и режет recall.

Вывод: retrieval-evidence в таком виде не использовать. Более перспективная ветка - factor-specific thresholds, реализованная в exp008.
