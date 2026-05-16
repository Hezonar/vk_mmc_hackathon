const fs = require("fs");

const a = JSON.parse(fs.readFileSync("reports/data_analysis/analysis.json", "utf8"));

function compact(ds) {
  return {
    rows: ds.rowCount,
    patients: ds.uniquePatients,
    date: `${ds.date.min}..${ds.date.max}`,
    target: ds.target,
    nestedItems: ds.nested.totalItems,
    nestedAvg: ds.nested.itemCountPerRow.mean,
    nestedEmptyRows: ds.nested.rowsWithEmptyList,
    nestedParseErrors: ds.nested.rowsWithParseError,
    factorAvg: ds.factorCountPerRow.mean,
    factorMax: ds.factorCountPerRow.max,
    missing: Object.fromEntries(Object.entries(ds.columns).map(([k, v]) => [k, `${v.missing} (${v.missingPct}%)`])),
    topHealth: ds.topLevelHealthGroups,
    topMkb: ds.topLevelMkbCodes.slice(0, 12),
    topFactors: ds.factors.slice(0, 12),
    topSpecialists: ds.nested.specialists.slice(0, 12),
    topNestedHealth: ds.nested.healthGroups,
    topNestedMkb: ds.nested.mkbCodes.slice(0, 12),
    highFactors: ds.targetRates.factors.slice(0, 12),
    highHealth: ds.targetRates.topLevelHealthGroups,
    highMkb: ds.targetRates.topLevelMkbCodes.slice(0, 12),
    highNestedMkb: ds.targetRates.nestedMkbCodes.slice(0, 12),
  };
}

console.log(JSON.stringify({
  train: compact(a.train),
  test: compact(a.test),
  comparison: a.comparison,
}, null, 2));
