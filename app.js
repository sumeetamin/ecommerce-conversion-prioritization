const byId = (id) => document.getElementById(id);
const pct = (value) => `${(value * 100).toFixed(1)}%`;
const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
let benchmark;

function renderThreshold() {
  const requested = Number(byId("threshold").value) / 100;
  const row = benchmark.test.threshold_curve.reduce((best, next) => Math.abs(next.threshold - requested) < Math.abs(best.threshold - requested) ? next : best);
  byId("threshold-out").textContent = pct(row.threshold);
  byId("threshold-metrics").innerHTML = [
    ["Sessions reviewed", `${row.sessions_reviewed.toLocaleString()} / ${benchmark.test.sessions.toLocaleString()}`],
    ["Precision", pct(row.precision)], ["Conversions captured", `${pct(row.conversion_capture_share)} (${row.tp.toLocaleString()})`],
    ["Recall", pct(row.recall)],
  ].map(([label, value]) => `<div class="threshold-metric"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("");
  drawCurve(row.threshold);
}

function drawCurve(threshold) {
  const svg = byId("pr-chart"); const width = Math.max(svg.clientWidth || 540, 320); const height = 250;
  const pad = { left: 44, right: 20, top: 18, bottom: 34 }; const plotW = width - pad.left - pad.right; const plotH = height - pad.top - pad.bottom;
  const points = benchmark.test.threshold_curve;
  const pathFor = (metric) => points.map((row, i) => `${i ? "L" : "M"}${pad.left + row.threshold * plotW},${pad.top + (1 - row[metric]) * plotH}`).join(" ");
  const selected = points.reduce((best, row) => Math.abs(row.threshold - threshold) < Math.abs(best.threshold - threshold) ? row : best);
  const x = pad.left + selected.threshold * plotW; const yP = pad.top + (1 - selected.precision) * plotH; const yR = pad.top + (1 - selected.recall) * plotH;
  let grid = "";
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + plotH * i / 4; const label = 100 - i * 25;
    grid += `<line class="axis" x1="${pad.left}" x2="${width-pad.right}" y1="${y}" y2="${y}"/><text class="chart-label" x="3" y="${y+3}">${label}%</text>`;
  }
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `${grid}<line class="axis" x1="${pad.left}" x2="${width-pad.right}" y1="${pad.top+plotH}" y2="${pad.top+plotH}"/><path class="curve-precision" d="${pathFor("precision")}"/><path class="curve-recall" d="${pathFor("recall")}"/><circle class="chart-point" cx="${x}" cy="${yP}" r="5"><title>Precision ${pct(selected.precision)} at ${pct(selected.threshold)}</title></circle><circle class="chart-point" cx="${x}" cy="${yR}" r="5"><title>Recall ${pct(selected.recall)} at ${pct(selected.threshold)}</title></circle><text class="chart-label" x="${pad.left}" y="${height-9}">LOWER SCORE THRESHOLD</text><text class="chart-label" text-anchor="end" x="${width-pad.right}" y="${height-9}">HIGHER →</text>`;
}

function render() {
  const test = benchmark.test; const selectedMetrics = test.metrics_at_validation_threshold; const protocol = benchmark.protocol;
  byId("headline-metrics").innerHTML = [
    ["Held-out sessions", test.sessions.toLocaleString(), "final test split"],
    ["Positive rate", pct(test.prevalence), "observed purchases"],
    ["Test average precision", pct(selectedMetrics.average_precision), `threshold ${pct(protocol.threshold)}`],
    ["Test ROC-AUC", pct(selectedMetrics.roc_auc), "rank discrimination"],
  ].map(([label, value, sub]) => `<div class="metric"><div class="metric-label">${esc(label)}</div><div class="metric-value">${esc(value)}</div><div class="metric-sub">${esc(sub)}</div></div>`).join("");
  const candidate = protocol.candidate_validation_metrics[protocol.selected_model]; const baselineValidation = protocol.candidate_validation_metrics.most_frequent_baseline;
  const baselineTest = test.majority_probability_baseline;
  byId("model-rows").innerHTML = `<tr class="selected-row"><td>${esc(protocol.selected_model.replaceAll("_", " "))} · selected</td><td>${pct(candidate.average_precision)}</td><td>${pct(selectedMetrics.average_precision)}</td><td>${selectedMetrics.brier_score.toFixed(4)}</td></tr><tr><td>Majority-probability baseline</td><td>${pct(baselineValidation.average_precision)}</td><td>${pct(baselineTest.average_precision)}</td><td>${baselineTest.brier_score.toFixed(4)}</td></tr>`;
  const validationScores = Object.entries(protocol.candidate_validation_metrics).map(([name, score]) => `${name.replaceAll("_", " ")}: ${pct(score.average_precision)} AP`).join(" · ");
  byId("split-note").innerHTML = `<b>Protocol:</b> ${esc(protocol.split)}.<br><b>Validation average precision:</b> ${esc(validationScores)}.<br><b>Selected:</b> <code>${esc(protocol.selected_model)}</code>. The final test split is used once. Validation-selected F1 threshold: <code>${pct(protocol.threshold)}</code>.`;
  byId("segments").innerHTML = test.visitor_segments.map((s) => `<article class="segment-card"><b>${esc(s.visitor_type)}</b><p>${s.sessions.toLocaleString()} held-out sessions<br>Observed purchase rate: <strong>${pct(s.conversion_rate)}</strong><br>Mean predicted score: ${pct(s.mean_score)}</p></article>`).join("");
  byId("limitations").innerHTML = benchmark.limitations.map((item) => `<li>${esc(item)}</li>`).join("");
  byId("threshold").value = String(Math.round(protocol.threshold * 100));
  byId("threshold").addEventListener("input", renderThreshold);
  window.addEventListener("resize", () => drawCurve(Number(byId("threshold").value) / 100));
  renderThreshold();
}

fetch("data/benchmark.json").then((response) => { if (!response.ok) throw new Error(`Benchmark report returned ${response.status}`); return response.json(); })
  .then((data) => { benchmark = data; byId("load-state").hidden = true; byId("dashboard").hidden = false; render(); })
  .catch((error) => { byId("load-state").textContent = `Could not load the benchmark report: ${error.message}. Open this demo through the portfolio site or a local HTTP server.`; });

