import React, { ChangeEvent, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Database,
  Download,
  FileText,
  Loader2,
  Play,
  Server,
  Settings2,
  Upload,
} from "lucide-react";
import "./styles.css";

type GapRow = {
  mapped_feature_name: string;
  cluster_label: string;
  visibility_share: number;
  consistency_band: string;
  target_visibility_status: string;
  competitor_present: boolean;
  is_feature_visibility_gap: boolean;
  gap_category: string;
  gap_severity: string;
  gap_reason: string;
  signal: string;
  prompt_count: number;
  top_competitor_brand_name: string;
  top_competitor_visibility_share: number;
  top_query: string;
  example_queries: string;
  present_prompt_ids: string;
  missing_prompt_ids: string;
};

type ApiResult = {
  ok: boolean;
  error?: string;
  stdout?: string;
  stderr?: string;
  overview: GapRow[];
  details: Record<string, unknown>[];
  coverage: Record<string, unknown>[];
  summary: string;
  metadata: Record<string, unknown>;
  extracted_features?: Record<string, unknown>[];
  extracted_text_preview?: string;
};

type Mode = "openai_mock" | "heuristic" | "openai";
type BrandMode = "openai_mock" | "keyword" | "openai";

const severityOrder: Record<string, number> = { high: 0, medium: 1, low: 2 };

function isoDateDaysAgo(daysAgo: number) {
  const date = new Date();
  date.setDate(date.getDate() - daysAgo);
  return date.toISOString().slice(0, 10);
}

function percent(value: number | string | undefined) {
  const parsed = Number(value ?? 0);
  if (!Number.isFinite(parsed)) return "0.0%";
  return `${(parsed * 100).toFixed(1)}%`;
}

function fileLabel(file: File | null, fallback: string) {
  return file ? file.name : fallback;
}

function formatBytes(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function parseCsvLine(line: string) {
  const values: string[] = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"' && line[index + 1] === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      values.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current.trim());
  return values;
}

function extractBrandNames(csvText: string) {
  const lines = csvText.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) return [];
  const headers = parseCsvLine(lines[0]).map((header) => header.replace(/^"|"$/g, "").trim().toLowerCase());
  const brandIndex = ["brand_name", "brand", "name"].map((name) => headers.indexOf(name)).find((index) => index >= 0);
  if (brandIndex === undefined) return [];
  return Array.from(
    new Set(
      lines
        .slice(1)
        .map((line) => parseCsvLine(line)[brandIndex]?.replace(/^"|"$/g, "").trim())
        .filter(Boolean),
    ),
  );
}

function CsvDownload({ rows, filename }: { rows: Record<string, unknown>[]; filename: string }) {
  const href = useMemo(() => {
    if (!rows.length) return "";
    const headers = Object.keys(rows[0]);
    const escape = (value: unknown) => `"${String(value ?? "").replaceAll('"', '""')}"`;
    const csv = [headers.join(","), ...rows.map((row) => headers.map((header) => escape(row[header])).join(","))].join("\n");
    return URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  }, [rows]);

  if (!href) return null;
  return (
    <a className="button button-secondary" href={href} download={filename}>
      <Download size={16} />
      Export CSV
    </a>
  );
}

function FilePicker({
  label,
  accept,
  file,
  onChange,
}: {
  label: string;
  accept: string;
  file: File | null;
  onChange: (file: File | null) => void;
}) {
  return (
    <label className="file-picker">
      <input
        type="file"
        accept={accept}
        onChange={(event: ChangeEvent<HTMLInputElement>) => onChange(event.target.files?.[0] ?? null)}
      />
      <span className="file-icon">
        <Upload size={17} />
      </span>
      <span>
        <strong>{label}</strong>
        <small>{file ? `${fileLabel(file, "Choose file")} · ${formatBytes(file.size)}` : "Choose file"}</small>
      </span>
    </label>
  );
}

function SelectControl({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function GapCard({ row, selected, onSelect }: { row: GapRow; selected: boolean; onSelect: () => void }) {
  const visible = Math.max(0, Math.min(100, Number(row.visibility_share || 0) * 100));
  return (
    <button className={`gap-card ${selected ? "selected" : ""}`} onClick={onSelect}>
      <div className="gap-card-top">
        <span className={`severity severity-${row.gap_severity}`}>{row.gap_severity}</span>
        <span className="strict-pill">{row.is_feature_visibility_gap ? "strict gap" : row.gap_category}</span>
      </div>
      <h3>{row.mapped_feature_name}</h3>
      <p>{row.cluster_label}</p>
      <div className="bar-row">
        <span>{percent(row.visibility_share)}</span>
        <div className="bar-track">
          <div className="bar-fill" style={{ width: `${visible}%` }} />
        </div>
      </div>
      <div className="gap-meta">
        <span>{row.prompt_count} prompts</span>
        <span>{row.top_competitor_brand_name || "No competitor"}</span>
      </div>
    </button>
  );
}

function App() {
  const [promptsCsv, setPromptsCsv] = useState<File | null>(null);
  const [brandsCsv, setBrandsCsv] = useState<File | null>(null);
  const [featureFile, setFeatureFile] = useState<File | null>(null);
  const [dataSource, setDataSource] = useState<"peec" | "csv">("peec");
  const [projectId, setProjectId] = useState("");
  const [startDate, setStartDate] = useState(isoDateDaysAgo(7));
  const [endDate, setEndDate] = useState(isoDateDaysAgo(0));
  const [targetBrand, setTargetBrand] = useState("Peec AI");
  const [normalizer, setNormalizer] = useState<Mode>("openai_mock");
  const [brandDetector, setBrandDetector] = useState<BrandMode>("openai_mock");
  const [embeddingBackend, setEmbeddingBackend] = useState("hash");
  const [aggregationMode, setAggregationMode] = useState("prompt");
  const [featureMode, setFeatureMode] = useState("mock");
  const [brandOptions, setBrandOptions] = useState<string[]>(["Peec AI"]);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [error, setError] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [resultSource, setResultSource] = useState<"sample" | "peec" | "csv" | null>(null);
  const [resultsMode, setResultsMode] = useState<"gaps" | "all">("all");
  const [settingsOpen, setSettingsOpen] = useState(false);

  const sortedOverview = useMemo(() => {
    const rows = [...(result?.overview ?? [])];
    return rows.sort((a, b) => {
      const strictDelta = Number(b.is_feature_visibility_gap) - Number(a.is_feature_visibility_gap);
      if (strictDelta) return strictDelta;
      const severityDelta = (severityOrder[a.gap_severity] ?? 9) - (severityOrder[b.gap_severity] ?? 9);
      if (severityDelta) return severityDelta;
      return Number(a.visibility_share) - Number(b.visibility_share);
    });
  }, [result]);

  const strictGapCount = sortedOverview.filter((row) => row.is_feature_visibility_gap).length;
  const avgVisibility = sortedOverview.length
    ? sortedOverview.reduce((sum, row) => sum + Number(row.visibility_share || 0), 0) / sortedOverview.length
    : 0;
  const featureCount = new Set(sortedOverview.map((row) => row.mapped_feature_name)).size;
  const visibleRows = resultsMode === "gaps" ? sortedOverview.filter((row) => row.is_feature_visibility_gap) : sortedOverview;
  const selected = visibleRows[Math.min(selectedIndex, Math.max(visibleRows.length - 1, 0))];
  const peecReady = Boolean(startDate && endDate && targetBrand && featureFile);
  const csvReady = Boolean(promptsCsv && brandsCsv && targetBrand && featureFile);
  const uploadReady = dataSource === "peec" ? peecReady : csvReady;
  const realModeSelected = normalizer === "openai" || brandDetector === "openai" || featureMode === "openai";

  useEffect(() => {
    let cancelled = false;
    fetch("/api/health")
      .then((response) => {
        if (!cancelled) setApiOnline(response.ok);
      })
      .catch(() => {
        if (!cancelled) setApiOnline(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!brandsCsv) {
      setBrandOptions([]);
      return;
    }
    let cancelled = false;
    brandsCsv.text().then((text) => {
      if (cancelled) return;
      const names = extractBrandNames(text);
      if (!names.length) return;
      setBrandOptions(names);
      if (!names.includes(targetBrand)) {
        setTargetBrand(names[0]);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [brandsCsv]);

  async function analyzeSample() {
    setLoading(true);
    setError("");
    setSelectedIndex(0);
    setTargetBrand("Peec AI");
    try {
      const response = await fetch("/api/analyze-sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_brand: "Peec AI", normalizer, brand_detector: brandDetector, embedding_backend: embeddingBackend, aggregation_mode: aggregationMode }),
      });
      const data = (await response.json()) as ApiResult;
      if (!response.ok || !data.ok) throw new Error(data.error || data.stderr || "Sample analysis failed.");
      setResult(data);
      setResultSource("sample");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sample analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  async function analyzeUpload() {
    setLoading(true);
    setError("");
    setSelectedIndex(0);
    try {
      if (!featureFile) throw new Error("Feature CSV or PDF is required.");

      const form = new FormData();
      let endpoint = "/api/analyze-peec";
      if (dataSource === "csv") {
        if (!promptsCsv || !brandsCsv) throw new Error("CSV fallback requires Prompts CSV and Brands CSV.");
        form.append("prompts_csv", promptsCsv);
        form.append("brands_csv", brandsCsv);
        endpoint = "/api/analyze";
      } else {
        form.append("project_id", projectId);
        form.append("start_date", startDate);
        form.append("end_date", endDate);
        if (brandsCsv) form.append("brands_csv", brandsCsv);
      }
      form.append("feature_file", featureFile);
      form.append("target_brand", targetBrand);
      form.append("normalizer", normalizer);
      form.append("brand_detector", brandDetector);
      form.append("embedding_backend", embeddingBackend);
      form.append("aggregation_mode", aggregationMode);
      form.append("feature_mode", featureMode);

      const response = await fetch(endpoint, { method: "POST", body: form });
      const data = (await response.json()) as ApiResult;
      if (!response.ok || !data.ok) throw new Error(data.error || data.stderr || "Analysis failed.");
      setResult(data);
      setResultSource(dataSource);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <button className="icon-button settings-trigger" onClick={() => setSettingsOpen((value) => !value)} title="Run settings">
        <Settings2 size={18} />
      </button>

      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="mark">
            <BarChart3 size={22} />
          </div>
          <div>
            <h1>Feature Visibility</h1>
            <p>AI answer gap analysis</p>
          </div>
        </div>

        <div className={`service-card ${apiOnline ? "online" : apiOnline === false ? "offline" : ""}`}>
          <Server size={17} />
          <span>{apiOnline === null ? "Checking API" : apiOnline ? "API connected" : "API unavailable"}</span>
        </div>

        <section className="panel">
          <div className="panel-title">
            <Database size={16} />
            <h2>Inputs</h2>
          </div>
          <div className="segment">
            <button className={dataSource === "peec" ? "active" : ""} onClick={() => setDataSource("peec")}>Peec MCP</button>
            <button className={dataSource === "csv" ? "active" : ""} onClick={() => setDataSource("csv")}>CSV fallback</button>
          </div>
          {dataSource === "peec" ? (
            <div className="compact-fields">
              <label className="field">
                <span>Peec project ID (optional)</span>
                <input
                  value={projectId}
                  placeholder="Defaults to first accessible Peec project"
                  onChange={(event) => setProjectId(event.target.value)}
                />
              </label>
              <div className="two-col">
                <label className="field">
                  <span>Start date</span>
                  <input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
                </label>
                <label className="field">
                  <span>End date</span>
                  <input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} />
                </label>
              </div>
            </div>
          ) : (
            <>
              <FilePicker label="Prompts CSV" accept=".csv" file={promptsCsv} onChange={setPromptsCsv} />
              <FilePicker label="Brands CSV" accept=".csv" file={brandsCsv} onChange={setBrandsCsv} />
            </>
          )}
          <FilePicker label="Feature CSV or PDF" accept=".csv,.pdf" file={featureFile} onChange={setFeatureFile} />
          {dataSource === "peec" ? <FilePicker label="Brands CSV override" accept=".csv" file={brandsCsv} onChange={setBrandsCsv} /> : null}
          {dataSource === "peec" ? <div className="notice">Prompts and responses come from Peec MCP. If project ID is empty, the first accessible Peec project is used. Brand CSV is optional.</div> : null}
        </section>

        <div className="actions">
          <button className="button button-primary" onClick={analyzeUpload} disabled={loading || !uploadReady || apiOnline === false}>
            {loading ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
            Run analysis
          </button>
          <button className="button button-secondary" onClick={analyzeSample} disabled={loading || apiOnline === false}>
            <FileText size={17} />
            Load sample
          </button>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Target brand</p>
            {brandOptions.length ? (
              <select className="top-select" value={targetBrand} onChange={(event) => setTargetBrand(event.target.value)}>
                {brandOptions.map((name) => <option key={name} value={name}>{name}</option>)}
              </select>
            ) : (
              <input className="top-input" value={targetBrand} onChange={(event) => setTargetBrand(event.target.value)} />
            )}
          </div>
          <div className="top-actions">
            <div className="status-pill">
              {loading ? <Loader2 className="spin" size={16} /> : result?.ok ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
              {loading ? "Running analysis" : result?.ok ? `Analysis ready · ${resultSource}` : uploadReady ? "Ready to run" : "Waiting for inputs"}
            </div>
          </div>
        </header>

        {settingsOpen ? (
          <section className="settings-popover">
            <SelectControl label="Prompt normalizer" value={normalizer} options={["openai_mock", "heuristic", "openai"]} onChange={(value) => setNormalizer(value as Mode)} />
            <SelectControl label="Brand detector" value={brandDetector} options={["openai_mock", "keyword", "openai"]} onChange={(value) => setBrandDetector(value as BrandMode)} />
            <SelectControl label="PDF extraction" value={featureMode} options={["mock", "openai"]} onChange={setFeatureMode} />
            <SelectControl label="Embeddings" value={embeddingBackend} options={["hash", "bge-m3"]} onChange={setEmbeddingBackend} />
            <SelectControl label="Aggregation" value={aggregationMode} options={["response", "prompt", "prompt_model"]} onChange={setAggregationMode} />
            {realModeSelected ? <div className="notice">Real LLM modes require `OPENAI_API_KEY` in the API server environment.</div> : null}
          </section>
        ) : null}

        {error ? <div className="error-banner">{error}</div> : null}
        {!uploadReady && !result ? (
          <div className="setup-banner">{dataSource === "peec" ? "Choose a date range and upload one feature CSV or PDF. Project ID is optional, or run the built-in sample." : "Upload prompts, brands, and one feature CSV or PDF, or run the built-in sample."}</div>
        ) : null}

        <section className="kpi-grid">
          <div className="kpi"><span>Strict gaps</span><strong>{strictGapCount}</strong></div>
          <div className="kpi"><span>Average visibility</span><strong>{percent(avgVisibility)}</strong></div>
          <div className="kpi"><span>Features</span><strong>{featureCount}</strong></div>
          <div className="kpi"><span>Rows analyzed</span><strong>{result?.metadata?.brand_count ? String(result.metadata.brand_count) : "-"}</strong></div>
        </section>

        <div className="content-grid">
          <section className="results-column">
            <div className="section-heading">
              <div>
                <h2>Ranked gaps</h2>
                <p>{visibleRows.length} visible rows</p>
              </div>
              <div className="toolbar">
                <div className="mini-segment">
                  <button className={resultsMode === "all" ? "active" : ""} onClick={() => setResultsMode("all")}>All</button>
                  <button className={resultsMode === "gaps" ? "active" : ""} onClick={() => setResultsMode("gaps")}>Strict gaps</button>
                </div>
                <CsvDownload rows={sortedOverview as unknown as Record<string, unknown>[]} filename="feature_gap_overview.csv" />
              </div>
            </div>
            {loading ? (
              <div className="empty-state"><Loader2 className="spin" size={28} />Running analysis</div>
            ) : visibleRows.length ? (
              <div className="gap-list">
                {visibleRows.map((row, index) => (
                  <GapCard key={`${row.mapped_feature_name}-${row.cluster_label}-${index}`} row={row} selected={index === selectedIndex} onSelect={() => setSelectedIndex(index)} />
                ))}
              </div>
            ) : (
              <div className="empty-state">{result ? "No rows match the current filter." : "Run a sample or upload files to see feature visibility gaps."}</div>
            )}
          </section>

          <section className="detail-panel">
            <div className="section-heading">
              <h2>Detail</h2>
            </div>
            {selected ? (
              <div className="detail-stack">
                <div>
                  <p className="eyebrow">Feature</p>
                  <h3>{selected.mapped_feature_name}</h3>
                  <p>{selected.cluster_label}</p>
                </div>
                <div className="detail-metrics">
                  <div><span>Visibility</span><strong>{percent(selected.visibility_share)}</strong></div>
                  <div><span>Status</span><strong>{selected.target_visibility_status}</strong></div>
                  <div><span>Competitor</span><strong>{selected.top_competitor_brand_name || "-"}</strong></div>
                </div>
                <div className="reason-box">{selected.gap_reason}</div>
                <div>
                  <p className="eyebrow">Top query</p>
                  <p>{selected.top_query || "-"}</p>
                </div>
                <div>
                  <p className="eyebrow">Example queries</p>
                  <ul className="query-list">
                    {(selected.example_queries || "").split(";").filter(Boolean).slice(0, 5).map((query) => (
                      <li key={query}>{query}</li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : (
              <div className="empty-state">Select a gap to inspect.</div>
            )}
          </section>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
