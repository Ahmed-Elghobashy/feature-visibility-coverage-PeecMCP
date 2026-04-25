import React, { ChangeEvent, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Download,
  FileText,
  Loader2,
  Play,
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

function percent(value: number | string | undefined) {
  const parsed = Number(value ?? 0);
  if (!Number.isFinite(parsed)) return "0.0%";
  return `${(parsed * 100).toFixed(1)}%`;
}

function fileLabel(file: File | null, fallback: string) {
  return file ? file.name : fallback;
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
        <small>{fileLabel(file, "Choose file")}</small>
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
  const [featuresCsv, setFeaturesCsv] = useState<File | null>(null);
  const [featurePdf, setFeaturePdf] = useState<File | null>(null);
  const [featureSource, setFeatureSource] = useState<"csv" | "pdf">("csv");
  const [targetBrand, setTargetBrand] = useState("Peec AI");
  const [normalizer, setNormalizer] = useState<Mode>("openai_mock");
  const [brandDetector, setBrandDetector] = useState<BrandMode>("openai_mock");
  const [embeddingBackend, setEmbeddingBackend] = useState("hash");
  const [aggregationMode, setAggregationMode] = useState("prompt");
  const [featureMode, setFeatureMode] = useState("mock");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [error, setError] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);

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

  const selected = sortedOverview[Math.min(selectedIndex, Math.max(sortedOverview.length - 1, 0))];
  const strictGapCount = sortedOverview.filter((row) => row.is_feature_visibility_gap).length;
  const avgVisibility = sortedOverview.length
    ? sortedOverview.reduce((sum, row) => sum + Number(row.visibility_share || 0), 0) / sortedOverview.length
    : 0;
  const featureCount = new Set(sortedOverview.map((row) => row.mapped_feature_name)).size;

  async function analyzeSample() {
    setLoading(true);
    setError("");
    setSelectedIndex(0);
    try {
      const response = await fetch("/api/analyze-sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_brand: targetBrand, normalizer, brand_detector: brandDetector, embedding_backend: embeddingBackend, aggregation_mode: aggregationMode }),
      });
      const data = (await response.json()) as ApiResult;
      if (!response.ok || !data.ok) throw new Error(data.error || data.stderr || "Sample analysis failed.");
      setResult(data);
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
      if (!promptsCsv || !brandsCsv) throw new Error("Prompts CSV and Brands CSV are required.");
      if (featureSource === "csv" && !featuresCsv) throw new Error("Features CSV is required.");
      if (featureSource === "pdf" && !featurePdf) throw new Error("Feature PDF is required.");

      const form = new FormData();
      form.append("prompts_csv", promptsCsv);
      form.append("brands_csv", brandsCsv);
      if (featureSource === "csv" && featuresCsv) form.append("features_csv", featuresCsv);
      if (featureSource === "pdf" && featurePdf) form.append("feature_pdf", featurePdf);
      form.append("target_brand", targetBrand);
      form.append("normalizer", normalizer);
      form.append("brand_detector", brandDetector);
      form.append("embedding_backend", embeddingBackend);
      form.append("aggregation_mode", aggregationMode);
      form.append("feature_mode", featureMode);

      const response = await fetch("/api/analyze", { method: "POST", body: form });
      const data = (await response.json()) as ApiResult;
      if (!response.ok || !data.ok) throw new Error(data.error || data.stderr || "Analysis failed.");
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
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

        <section className="panel">
          <h2>Inputs</h2>
          <FilePicker label="Prompts CSV" accept=".csv" file={promptsCsv} onChange={setPromptsCsv} />
          <FilePicker label="Brands CSV" accept=".csv" file={brandsCsv} onChange={setBrandsCsv} />
          <div className="segment">
            <button className={featureSource === "csv" ? "active" : ""} onClick={() => setFeatureSource("csv")}>CSV</button>
            <button className={featureSource === "pdf" ? "active" : ""} onClick={() => setFeatureSource("pdf")}>PDF</button>
          </div>
          {featureSource === "csv" ? (
            <FilePicker label="Features CSV" accept=".csv" file={featuresCsv} onChange={setFeaturesCsv} />
          ) : (
            <>
              <FilePicker label="Feature PDF" accept=".pdf" file={featurePdf} onChange={setFeaturePdf} />
              <SelectControl label="PDF extraction" value={featureMode} options={["mock", "openai"]} onChange={setFeatureMode} />
            </>
          )}
        </section>

        <section className="panel">
          <h2>Run settings</h2>
          <label className="field">
            <span>Target brand</span>
            <input value={targetBrand} onChange={(event) => setTargetBrand(event.target.value)} />
          </label>
          <SelectControl label="Prompt normalizer" value={normalizer} options={["openai_mock", "heuristic", "openai"]} onChange={(value) => setNormalizer(value as Mode)} />
          <SelectControl label="Brand detector" value={brandDetector} options={["openai_mock", "keyword", "openai"]} onChange={(value) => setBrandDetector(value as BrandMode)} />
          <SelectControl label="Embeddings" value={embeddingBackend} options={["hash", "bge-m3"]} onChange={setEmbeddingBackend} />
          <SelectControl label="Aggregation" value={aggregationMode} options={["response", "prompt", "prompt_model"]} onChange={setAggregationMode} />
        </section>

        <div className="actions">
          <button className="button button-primary" onClick={analyzeUpload} disabled={loading}>
            {loading ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
            Run analysis
          </button>
          <button className="button button-secondary" onClick={analyzeSample} disabled={loading}>
            <FileText size={17} />
            Load sample
          </button>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Target brand</p>
            <h2>{targetBrand || "Not selected"}</h2>
          </div>
          <div className="status-pill">
            {result?.ok ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
            {result?.ok ? "Analysis ready" : "Waiting for run"}
          </div>
        </header>

        {error ? <div className="error-banner">{error}</div> : null}

        <section className="kpi-grid">
          <div className="kpi"><span>Strict gaps</span><strong>{strictGapCount}</strong></div>
          <div className="kpi"><span>Average visibility</span><strong>{percent(avgVisibility)}</strong></div>
          <div className="kpi"><span>Features</span><strong>{featureCount}</strong></div>
          <div className="kpi"><span>Rows analyzed</span><strong>{result?.metadata?.brand_count ? String(result.metadata.brand_count) : "-"}</strong></div>
        </section>

        <div className="content-grid">
          <section className="results-column">
            <div className="section-heading">
              <h2>Ranked gaps</h2>
              <CsvDownload rows={sortedOverview as unknown as Record<string, unknown>[]} filename="feature_gap_overview.csv" />
            </div>
            {loading ? (
              <div className="empty-state"><Loader2 className="spin" size={28} />Running analysis</div>
            ) : sortedOverview.length ? (
              <div className="gap-list">
                {sortedOverview.map((row, index) => (
                  <GapCard key={`${row.mapped_feature_name}-${row.cluster_label}-${index}`} row={row} selected={index === selectedIndex} onSelect={() => setSelectedIndex(index)} />
                ))}
              </div>
            ) : (
              <div className="empty-state">Run a sample or upload files to see feature visibility gaps.</div>
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
