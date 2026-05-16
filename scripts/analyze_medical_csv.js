const fs = require("fs");
const path = require("path");

const root = process.cwd();
const dataDir = path.join(root, "data");
const outDir = path.join(root, "reports", "data_analysis");
fs.mkdirSync(outDir, { recursive: true });

function parseCsv(text, onRecord) {
  let field = "";
  let record = [];
  let inQuotes = false;

  function pushRecord() {
    if (record.length === 1 && record[0] === "" && field === "") return;
    record.push(field);
    onRecord(record);
    record = [];
    field = "";
  }

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      record.push(field);
      field = "";
    } else if (ch === "\n") {
      if (field.endsWith("\r")) field = field.slice(0, -1);
      pushRecord();
    } else {
      field += ch;
    }
  }
  if (field.length || record.length) pushRecord();
}

function topEntries(map, limit = 20) {
  return [...map.entries()]
    .sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0]), "ru"))
    .slice(0, limit)
    .map(([value, count]) => ({ value, count }));
}

function bump(map, key, inc = 1) {
  const value = key === undefined || key === null ? "" : String(key).trim();
  if (!value) return;
  map.set(value, (map.get(value) || 0) + inc);
}

function bumpRate(map, key, positive) {
  const value = key === undefined || key === null ? "" : String(key).trim();
  if (!value) return;
  const item = map.get(value) || { total: 0, positive: 0 };
  item.total += 1;
  if (positive) item.positive += 1;
  map.set(value, item);
}

function topRates(map, minTotal = 30, limit = 30) {
  return [...map.entries()]
    .map(([value, x]) => ({ value, total: x.total, positive: x.positive, positivePct: pct(x.positive, x.total) }))
    .filter((x) => x.total >= minTotal)
    .sort((a, b) => b.positivePct - a.positivePct || b.positive - a.positive || b.total - a.total)
    .slice(0, limit);
}

function pct(n, d) {
  return d ? +(100 * n / d).toFixed(2) : 0;
}

function parseDate(value) {
  if (!value) return null;
  const m = value.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!m) return null;
  return `${m[1]}-${m[2]}-${m[3]}`;
}

function splitFactors(value) {
  if (!value) return [];
  return value
    .split(";")
    .map((x) => x.trim())
    .filter(Boolean);
}

function mkColumnStats(headers) {
  return Object.fromEntries(headers.map((h) => [h, {
    missing: 0,
    nonMissing: 0,
    unique: new Set(),
    uniqueOverflow: false,
    top: new Map(),
    minLen: Infinity,
    maxLen: 0,
    totalLen: 0,
  }]));
}

function observeColumn(stat, value) {
  const v = value == null ? "" : String(value);
  if (v.trim() === "") {
    stat.missing += 1;
    return;
  }
  stat.nonMissing += 1;
  stat.minLen = Math.min(stat.minLen, v.length);
  stat.maxLen = Math.max(stat.maxLen, v.length);
  stat.totalLen += v.length;
  if (stat.unique.size < 50000) stat.unique.add(v);
  else stat.uniqueOverflow = true;
  if (stat.top.size < 20000 || stat.top.has(v)) bump(stat.top, v);
}

function analyzeFile(fileName) {
  const filePath = path.join(dataDir, fileName);
  const text = fs.readFileSync(filePath, "utf8").replace(/^\uFEFF/, "");
  let headers = null;
  let columns = null;
  const rows = [];

  const result = {
    file: fileName,
    bytes: fs.statSync(filePath).size,
    rowCount: 0,
    columnCount: 0,
    columns: {},
    date: { min: null, max: null, byMonth: new Map() },
    patients: new Set(),
    examRows: new Set(),
    factors: new Map(),
    factorCountPerRow: [],
    nested: {
      rowsWithParseError: 0,
      rowsWithEmptyList: 0,
      totalItems: 0,
      itemCountPerRow: [],
      specialists: new Map(),
      healthGroups: new Map(),
      mkbCodes: new Map(),
      mkbDescriptions: new Map(),
      conclusionEmpty: 0,
      conclusionLength: [],
    },
    target: new Map(),
    topLevelHealthGroups: new Map(),
    topLevelMkbCodes: new Map(),
    contraindicatedFactors: new Map(),
    targetRates: {
      factors: new Map(),
      topLevelHealthGroups: new Map(),
      topLevelMkbCodes: new Map(),
      nestedSpecialists: new Map(),
      nestedHealthGroups: new Map(),
      nestedMkbCodes: new Map(),
    },
    examples: [],
  };

  parseCsv(text, (record) => {
    if (!headers) {
      headers = record;
      result.columnCount = headers.length;
      columns = mkColumnStats(headers);
      return;
    }
    if (record.length !== headers.length) {
      result.badRecordCount = (result.badRecordCount || 0) + 1;
      return;
    }
    const row = Object.fromEntries(headers.map((h, i) => [h, record[i]]));
    result.rowCount += 1;
    if (result.examples.length < 3) result.examples.push(row);

    for (const h of headers) observeColumn(columns[h], row[h]);

    if (row.patient_id) result.patients.add(row.patient_id);
    if (row.exam_row_id) result.examRows.add(row.exam_row_id);
    const hasTarget = row.has_contraindications !== undefined && row.has_contraindications !== "";
    if (hasTarget) bump(result.target, row.has_contraindications);
    const positive = String(row.has_contraindications).toLowerCase() === "true";
    if (row.health_group) bump(result.topLevelHealthGroups, row.health_group);
    if (hasTarget && row.health_group) bumpRate(result.targetRates.topLevelHealthGroups, row.health_group, positive);
    if (row.mkb_code) bump(result.topLevelMkbCodes, row.mkb_code);
    if (hasTarget && row.mkb_code) bumpRate(result.targetRates.topLevelMkbCodes, row.mkb_code, positive);
    if (row.contraindicated_factors) {
      for (const factor of splitFactors(row.contraindicated_factors)) bump(result.contraindicatedFactors, factor);
    }

    const day = parseDate(row.consultation_date);
    if (day) {
      result.date.min = result.date.min && result.date.min < day ? result.date.min : day;
      result.date.max = result.date.max && result.date.max > day ? result.date.max : day;
      bump(result.date.byMonth, day.slice(0, 7));
    }

    const factors = splitFactors(row.assigned_harmful_factors);
    result.factorCountPerRow.push(factors.length);
    for (const factor of factors) {
      bump(result.factors, factor);
      if (hasTarget) bumpRate(result.targetRates.factors, factor, positive);
    }

    if (row.specialist_conclusions !== undefined) {
      let items;
      try {
        items = row.specialist_conclusions.trim() ? JSON.parse(row.specialist_conclusions) : [];
      } catch {
        result.nested.rowsWithParseError += 1;
        items = [];
      }
      if (!items.length) result.nested.rowsWithEmptyList += 1;
      result.nested.totalItems += items.length;
      result.nested.itemCountPerRow.push(items.length);
      for (const item of items) {
        bump(result.nested.specialists, item.specialist);
        bump(result.nested.healthGroups, item.health_group);
        bump(result.nested.mkbCodes, item.mkb_code);
        bump(result.nested.mkbDescriptions, item.mkb_description);
        if (hasTarget) {
          bumpRate(result.targetRates.nestedSpecialists, item.specialist, positive);
          bumpRate(result.targetRates.nestedHealthGroups, item.health_group, positive);
          bumpRate(result.targetRates.nestedMkbCodes, item.mkb_code, positive);
        }
        const c = item.conclusion ? String(item.conclusion).trim() : "";
        if (!c) result.nested.conclusionEmpty += 1;
        else result.nested.conclusionLength.push(c.length);
      }
    }
  });

  function summarizeArray(values) {
    if (!values.length) return { min: 0, p25: 0, median: 0, mean: 0, p75: 0, max: 0 };
    const sorted = [...values].sort((a, b) => a - b);
    const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
    return {
      min: sorted[0],
      p25: q(0.25),
      median: q(0.5),
      mean: +(values.reduce((a, b) => a + b, 0) / values.length).toFixed(2),
      p75: q(0.75),
      max: sorted[sorted.length - 1],
    };
  }

  result.uniquePatients = result.patients.size;
  result.uniqueExamRows = result.examRows.size;
  delete result.patients;
  delete result.examRows;
  result.factorCountPerRow = summarizeArray(result.factorCountPerRow);
  result.nested.itemCountPerRow = summarizeArray(result.nested.itemCountPerRow);
  result.nested.conclusionLength = summarizeArray(result.nested.conclusionLength);

  for (const [name, stat] of Object.entries(columns)) {
    result.columns[name] = {
      missing: stat.missing,
      missingPct: pct(stat.missing, result.rowCount),
      nonMissing: stat.nonMissing,
      unique: stat.uniqueOverflow ? `${stat.unique.size}+` : stat.unique.size,
      minLen: stat.minLen === Infinity ? 0 : stat.minLen,
      avgLen: stat.nonMissing ? +(stat.totalLen / stat.nonMissing).toFixed(1) : 0,
      maxLen: stat.maxLen,
      top: topEntries(stat.top, 10),
    };
  }

  result.allFactors = [...result.factors.keys()].sort();
  result.allNestedSpecialists = [...result.nested.specialists.keys()].sort();
  result.allNestedMkbCodes = [...result.nested.mkbCodes.keys()].sort();
  result.date.byMonth = topEntries(result.date.byMonth, 200).sort((a, b) => String(a.value).localeCompare(String(b.value)));
  result.factors = topEntries(result.factors, 50);
  result.target = topEntries(result.target, 10);
  result.topLevelHealthGroups = topEntries(result.topLevelHealthGroups, 20);
  result.topLevelMkbCodes = topEntries(result.topLevelMkbCodes, 30);
  result.contraindicatedFactors = topEntries(result.contraindicatedFactors, 50);
  result.nested.specialists = topEntries(result.nested.specialists, 50);
  result.nested.healthGroups = topEntries(result.nested.healthGroups, 20);
  result.nested.mkbCodes = topEntries(result.nested.mkbCodes, 50);
  result.nested.mkbDescriptions = topEntries(result.nested.mkbDescriptions, 50);
  result.targetRates.factors = topRates(result.targetRates.factors);
  result.targetRates.topLevelHealthGroups = topRates(result.targetRates.topLevelHealthGroups);
  result.targetRates.topLevelMkbCodes = topRates(result.targetRates.topLevelMkbCodes);
  result.targetRates.nestedSpecialists = topRates(result.targetRates.nestedSpecialists);
  result.targetRates.nestedHealthGroups = topRates(result.targetRates.nestedHealthGroups);
  result.targetRates.nestedMkbCodes = topRates(result.targetRates.nestedMkbCodes);
  return result;
}

function compareTrainTest(train, test) {
  const trainCols = new Set(Object.keys(train.columns));
  const testCols = new Set(Object.keys(test.columns));
  const onlyTrainColumns = [...trainCols].filter((x) => !testCols.has(x));
  const onlyTestColumns = [...testCols].filter((x) => !trainCols.has(x));
  const trainFactors = new Set(train.allFactors);
  const testFactors = new Set(test.allFactors);
  const trainSpecialists = new Set(train.allNestedSpecialists);
  const testSpecialists = new Set(test.allNestedSpecialists);
  const trainMkb = new Set(train.allNestedMkbCodes);
  const testMkb = new Set(test.allNestedMkbCodes);
  return {
    onlyTrainColumns,
    onlyTestColumns,
    factorOnlyInTest: [...testFactors].filter((x) => !trainFactors.has(x)).sort(),
    factorOnlyInTrain: [...trainFactors].filter((x) => !testFactors.has(x)).sort(),
    specialistOnlyInTest: [...testSpecialists].filter((x) => !trainSpecialists.has(x)).sort(),
    specialistOnlyInTrain: [...trainSpecialists].filter((x) => !testSpecialists.has(x)).sort(),
    nestedMkbOnlyInTest: [...testMkb].filter((x) => !trainMkb.has(x)).sort().slice(0, 100),
    nestedMkbOnlyInTrain: [...trainMkb].filter((x) => !testMkb.has(x)).sort().slice(0, 100),
  };
}

function barSvg(title, items, width = 900, height = 420) {
  const margin = { top: 42, right: 24, bottom: 42, left: 230 };
  const data = items.slice(0, 20);
  const rowH = Math.max(18, Math.floor((height - margin.top - margin.bottom) / Math.max(1, data.length)));
  const chartH = rowH * data.length;
  const max = Math.max(...data.map((d) => d.count), 1);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const bars = data.map((d, i) => {
    const y = margin.top + i * rowH;
    const w = Math.max(2, Math.round((width - margin.left - margin.right) * d.count / max));
    const label = esc(String(d.value).length > 34 ? `${String(d.value).slice(0, 33)}...` : d.value);
    return `<text x="${margin.left - 10}" y="${y + 13}" text-anchor="end">${label}</text><rect x="${margin.left}" y="${y}" width="${w}" height="${rowH - 5}" rx="3"/><text x="${margin.left + w + 6}" y="${y + 13}">${d.count}</text>`;
  }).join("\n");
  return `<svg viewBox="0 0 ${width} ${margin.top + chartH + margin.bottom}" role="img" aria-label="${esc(title)}"><style>svg{font-family:Arial,sans-serif} text{font-size:12px;fill:#1f2937} rect{fill:#2563eb}</style><text x="12" y="24" style="font-size:18px;font-weight:700">${esc(title)}</text>${bars}</svg>`;
}

function table(headers, rows) {
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  return `<table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((x) => `<td>${esc(x)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

const train = analyzeFile("train.csv");
const test = analyzeFile("test.csv");
const comparison = compareTrainTest(train, test);
const all = { generatedAt: new Date().toISOString(), train, test, comparison };
fs.writeFileSync(path.join(outDir, "analysis.json"), JSON.stringify(all, null, 2), "utf8");

function datasetSection(ds) {
  const colRows = Object.entries(ds.columns).map(([name, s]) => [name, s.missing, `${s.missingPct}%`, s.unique, s.avgLen, s.maxLen]);
  return `
    <section>
      <h2>${ds.file}</h2>
      <p><b>Строк:</b> ${ds.rowCount.toLocaleString("ru-RU")} · <b>колонок:</b> ${ds.columnCount} · <b>пациентов:</b> ${ds.uniquePatients.toLocaleString("ru-RU")} · <b>период:</b> ${ds.date.min} ... ${ds.date.max}</p>
      ${barSvg("Записи по месяцам", ds.date.byMonth)}
      ${ds.target.length ? barSvg("Целевая переменная has_contraindications", ds.target, 760, 160) : ""}
      ${barSvg("Топ вредных факторов", ds.factors)}
      ${ds.targetRates && ds.targetRates.factors.length ? barSvg("Вредные факторы с высокой долей противопоказаний", ds.targetRates.factors.map((x) => ({ value: `${x.value} (${x.positive}/${x.total})`, count: x.positivePct }))) : ""}
      ${barSvg("Топ специалистов во вложенных заключениях", ds.nested.specialists)}
      ${barSvg("Топ МКБ во вложенных заключениях", ds.nested.mkbCodes)}
      ${ds.topLevelHealthGroups.length ? barSvg("Группы здоровья верхнего уровня", ds.topLevelHealthGroups, 760, 260) : ""}
      <h3>Колонки: пропуски, уникальность, длины</h3>
      ${table(["Колонка", "Пропусков", "%", "Уникальных", "Средняя длина", "Макс. длина"], colRows)}
    </section>`;
}

const html = `<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Анализ train/test CSV</title>
  <style>
    body{font-family:Arial,sans-serif;margin:28px;color:#111827;background:#fff}
    h1,h2,h3{margin:22px 0 10px}
    section{border-top:1px solid #d1d5db;padding-top:18px}
    svg{max-width:100%;height:auto;margin:10px 0 18px;border:1px solid #e5e7eb}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 24px}
    th,td{border:1px solid #d1d5db;padding:6px 8px;text-align:left;vertical-align:top}
    th{background:#f3f4f6}
    code{background:#f3f4f6;padding:1px 4px;border-radius:3px}
  </style>
</head>
<body>
  <h1>Анализ медицинских CSV</h1>
  <p>Сгенерировано: ${all.generatedAt}. Полные численные данные лежат в <code>analysis.json</code>.</p>
  <section>
    <h2>Train/Test</h2>
    <p><b>Колонки только в train:</b> ${comparison.onlyTrainColumns.join(", ") || "нет"}.</p>
    <p><b>Колонки только в test:</b> ${comparison.onlyTestColumns.join(", ") || "нет"}.</p>
    <p><b>Факторы только в test:</b> ${comparison.factorOnlyInTest.join(", ") || "нет"}.</p>
    <p><b>Специалисты только в test:</b> ${comparison.specialistOnlyInTest.join(", ") || "нет"}.</p>
  </section>
  ${datasetSection(train)}
  ${datasetSection(test)}
</body>
</html>`;
fs.writeFileSync(path.join(outDir, "report.html"), html, "utf8");

const md = [
  "# Анализ data/train.csv и data/test.csv",
  "",
  `Сгенерировано: ${all.generatedAt}`,
  "",
  `## train.csv`,
  `- Строк: ${train.rowCount}; колонок: ${train.columnCount}; уникальных пациентов: ${train.uniquePatients}; период: ${train.date.min} ... ${train.date.max}.`,
  `- Цель has_contraindications: ${train.target.map((x) => `${x.value}=${x.count} (${pct(x.count, train.rowCount)}%)`).join(", ")}.`,
  `- Вложенных заключений специалистов: ${train.nested.totalItems}; среднее на запись: ${train.nested.itemCountPerRow.mean}; строк с пустым списком: ${train.nested.rowsWithEmptyList}.`,
  `- Топ вредных факторов: ${train.factors.slice(0, 10).map((x) => `${x.value}=${x.count}`).join(", ")}.`,
  `- Самая высокая доля противопоказаний по факторам (min 30 строк): ${train.targetRates.factors.slice(0, 10).map((x) => `${x.value}=${x.positivePct}% (${x.positive}/${x.total})`).join(", ")}.`,
  `- Топ специалистов: ${train.nested.specialists.slice(0, 10).map((x) => `${x.value}=${x.count}`).join(", ")}.`,
  `- Колонки только в train: ${comparison.onlyTrainColumns.join(", ")}.`,
  `- Колонки только в test: ${comparison.onlyTestColumns.join(", ") || "нет"}.`,
  "",
  `## test.csv`,
  `- Строк: ${test.rowCount}; колонок: ${test.columnCount}; уникальных пациентов: ${test.uniquePatients}; период: ${test.date.min} ... ${test.date.max}.`,
  `- Вложенных заключений специалистов: ${test.nested.totalItems}; среднее на запись: ${test.nested.itemCountPerRow.mean}; строк с пустым списком: ${test.nested.rowsWithEmptyList}.`,
  `- Топ вредных факторов: ${test.factors.slice(0, 10).map((x) => `${x.value}=${x.count}`).join(", ")}.`,
  `- Топ специалистов: ${test.nested.specialists.slice(0, 10).map((x) => `${x.value}=${x.count}`).join(", ")}.`,
  "",
  "## Артефакты",
  "- reports/data_analysis/report.html",
  "- reports/data_analysis/analysis.json",
].join("\n");
fs.writeFileSync(path.join(outDir, "summary.md"), md, "utf8");

console.log(JSON.stringify({
  report: path.join(outDir, "report.html"),
  summary: path.join(outDir, "summary.md"),
  json: path.join(outDir, "analysis.json"),
  trainRows: train.rowCount,
  testRows: test.rowCount,
  trainTarget: train.target,
}, null, 2));
