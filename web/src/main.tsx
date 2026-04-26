import React, { CSSProperties, ChangeEvent, useEffect, useMemo, useState } from "react";
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
  Upload,
} from "lucide-react";
import "./styles.css";

type GapRow = {
  mapped_feature_id?: string;
  mapped_feature_name: string;
  cluster_id?: string;
  cluster_label: string;
  visibility_share: number;
  consistency_band: string;
  target_visibility_status: string;
  target_feature_present_count?: number;
  target_feature_visible_count?: number;
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
  model_breakdown?: string;
  model_visibility_breakdown?: string;
  top_source_domains?: string;
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
  run_id?: string;
  manifest?: Record<string, unknown>;
};

type RunSummary = {
  run_id: string;
  created_at: string;
  source: string;
  target_brand: string;
  aggregation_mode: string;
  prompt_rows: number;
  overview_rows: number;
};

type MappingRow = {
  prompt_id: string;
  original_prompt?: string;
  canonical_query?: string;
  engine?: string;
  cluster_id?: string;
  cluster_label?: string;
  mapped_feature_id?: string;
  mapped_feature_name?: string;
  feature_similarity?: number;
  feature_present?: boolean;
  feature_evidence_strength?: number;
  source_domains?: string;
  brand_present?: boolean;
};

type ProgressEvent = {
  type: "run" | "stage" | "error" | "result";
  timestamp_ms: number;
  stage?: string;
  status?: string;
  message?: string;
  duration_ms?: number;
  result?: ApiResult;
  error?: string;
  prompt_rows?: number;
  feature_count?: number;
  brand_source?: string;
};

type RunLogEntry = {
  id: string;
  stage: string;
  status: string;
  message: string;
  timestampMs: number;
  durationMs?: number;
};

type Mode = "openai_mock" | "heuristic" | "openai";
type BrandMode = "openai_mock" | "keyword" | "openai";
type FeatureEvidenceMode = "openai_mock" | "keyword" | "openai";
type AppView = "setup" | "results";
type ResultsTab = "dashboard" | "history" | "features";

const severityOrder: Record<string, number> = { high: 0, medium: 1, low: 2 };
const clusterPalette = [
  { accent: "#9eb0cb", soft: "#f4f7fc", border: "#d8e0ee" },
  { accent: "#a8b8d2", soft: "#f5f8fd", border: "#dbe3f0" },
  { accent: "#93abc9", soft: "#f2f7fc", border: "#d3dfed" },
  { accent: "#b2bdd0", soft: "#f6f8fb", border: "#dfe5ed" },
  { accent: "#8ea4c0", soft: "#f2f6fb", border: "#d0dae7" },
  { accent: "#a6b6cc", soft: "#f5f8fc", border: "#d9e1eb" },
];

function isoDateDaysAgo(daysAgo: number) {
  const date = new Date();
  date.setDate(date.getDate() - daysAgo);
  return date.toISOString().slice(0, 10);
}

function formatRunDate(isoValue: string) {
  const date = new Date(isoValue);
  if (Number.isNaN(date.getTime())) return isoValue;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function formatRunLabel(run: RunSummary) {
  return `${run.target_brand} · ${formatRunDate(run.created_at)} · ${run.aggregation_mode}`;
}

function hashKey(value: string) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash;
}

function featureAccentStyle(featureKey: string | undefined, selected = false): CSSProperties {
  const key = featureKey || "default-feature";
  const palette = clusterPalette[hashKey(key) % clusterPalette.length];
  return {
    "--cluster-accent": palette.accent,
    "--cluster-soft": palette.soft,
    "--cluster-border": palette.border,
    "--cluster-shadow": selected ? `${palette.accent}22` : `${palette.accent}14`,
  } as CSSProperties;
}

type BreakdownItem = {
  label: string;
  value: number;
  displayValue: string;
};

function parseBreakdown(metric: string | undefined) {
  if (!metric) return [] as BreakdownItem[];
  return metric
    .split(";")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const parts = entry.split(":");
      if (parts.length < 2) return null;
      const label = parts.slice(0, -1).join(":").trim();
      const rawValue = parts[parts.length - 1].trim();
      const numeric = Number.parseFloat(rawValue.replace("%", ""));
      if (!label || !Number.isFinite(numeric)) return null;
      return {
        label,
        value: numeric,
        displayValue: rawValue.includes("%") ? rawValue : `${rawValue}`,
      };
    })
    .filter((item): item is BreakdownItem => Boolean(item));
}

function BreakdownChart({ title, items, suffix = "" }: { title: string; items: BreakdownItem[]; suffix?: string }) {
  if (!items.length) {
    return (
      <div>
        <p className="eyebrow">{title}</p>
        <p className="detail-muted">-</p>
      </div>
    );
  }
  const maxValue = Math.max(...items.map((item) => item.value), 1);
  return (
    <div>
      <p className="eyebrow">{title}</p>
      <div className="mini-chart">
        {items.map((item) => (
          <div key={`${title}-${item.label}`} className="mini-chart-row">
            <div className="mini-chart-meta">
              <span>{item.label}</span>
              <strong>{item.displayValue}{suffix}</strong>
            </div>
            <div className="mini-chart-track">
              <div className="mini-chart-fill" style={{ width: `${Math.max((item.value / maxValue) * 100, 6)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function trafficTone(value: string | undefined) {
  const normalized = String(value || "").toLowerCase();
  if (
    normalized.includes("missing")
    || normalized.includes("inconsistent")
    || normalized.includes("negative")
    || normalized.includes("weak")
  ) return "bad";
  if (
    normalized.includes("partial")
    || normalized.includes("medium")
    || normalized.includes("investigat")
    || normalized.includes("worth investigating")
  ) return "mid";
  if (normalized.includes("strong") || normalized.includes("good") || normalized.includes("visible")) return "good";
  return "neutral";
}

function sourceLabel(source: string) {
  if (source === "peec") return "Peec";
  if (source === "csv") return "CSV";
  if (source === "sample") return "Sample";
  if (source === "saved_run") return "Re-aggregated";
  return source;
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
  const accentStyle = featureAccentStyle(row.mapped_feature_id || row.mapped_feature_name, selected);
  return (
    <button className={`gap-card ${selected ? "selected" : ""}`} style={accentStyle} onClick={onSelect}>
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
  const [featureEvidenceMode, setFeatureEvidenceMode] = useState<FeatureEvidenceMode>("openai_mock");
  const [embeddingBackend, setEmbeddingBackend] = useState("hash");
  const [aggregationMode, setAggregationMode] = useState("prompt");
  const [featureMode, setFeatureMode] = useState("mock");
  const [peecLimit, setPeecLimit] = useState("250");
  const [brandOptions, setBrandOptions] = useState<string[]>(["Peec AI"]);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [error, setError] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [resultSource, setResultSource] = useState<"sample" | "peec" | "csv" | "saved_run" | null>(null);
  const [resultsMode, setResultsMode] = useState<"gaps" | "all">("all");
  const [workspaceView, setWorkspaceView] = useState<AppView>("setup");
  const [resultsTab, setResultsTab] = useState<ResultsTab>("dashboard");
  const [runLogs, setRunLogs] = useState<RunLogEntry[]>([]);
  const [activeStage, setActiveStage] = useState("");
  const [savedRuns, setSavedRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [runMappings, setRunMappings] = useState<MappingRow[]>([]);
  const [runMappingsLoading, setRunMappingsLoading] = useState(false);

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
  const selectedModelBreakdown = useMemo(
    () => parseBreakdown(selected?.model_visibility_breakdown || selected?.model_breakdown),
    [selected],
  );
  const selectedSourceBreakdown = useMemo(
    () => parseBreakdown(selected?.top_source_domains),
    [selected],
  );
  const selectedRunSummary = useMemo(() => savedRuns.find((run) => run.run_id === selectedRunId) || null, [savedRuns, selectedRunId]);
  const selectedComparison = useMemo(() => {
    if (!selected) return [] as BreakdownItem[];
    const items: BreakdownItem[] = [
      {
        label: selectedRunSummary?.target_brand || targetBrand || "Target",
        value: Number(selected.visibility_share || 0) * 100,
        displayValue: percent(selected.visibility_share),
      },
    ];
    if (selected.top_competitor_brand_name) {
      items.push({
        label: selected.top_competitor_brand_name,
        value: Number(selected.top_competitor_visibility_share || 0) * 100,
        displayValue: percent(selected.top_competitor_visibility_share),
      });
    }
    return items;
  }, [selected, selectedRunSummary, targetBrand]);
  const selectedDetailStyle = useMemo(
    () => featureAccentStyle(selected?.mapped_feature_id || selected?.mapped_feature_name, true),
    [selected],
  );
  const groupedRunMappings = useMemo(() => {
    const groups = new Map<string, { featureName: string; queries: MappingRow[] }>();
    for (const row of runMappings) {
      const featureName = row.mapped_feature_name || "Unmapped";
      const key = row.mapped_feature_id || featureName;
      if (!groups.has(key)) {
        groups.set(key, { featureName, queries: [] });
      }
      groups.get(key)!.queries.push(row);
    }
    return Array.from(groups.values())
      .map((group) => ({
        ...group,
        queries: group.queries
          .sort((left, right) => String(left.canonical_query || left.original_prompt || "").localeCompare(String(right.canonical_query || right.original_prompt || ""))),
      }))
      .sort((left, right) => right.queries.length - left.queries.length || left.featureName.localeCompare(right.featureName));
  }, [runMappings]);
  const peecReady = Boolean(startDate && endDate && targetBrand && featureFile);
  const csvReady = Boolean(promptsCsv && brandsCsv && targetBrand && featureFile);
  const uploadReady = dataSource === "peec" ? peecReady : csvReady;
  const realModeSelected = normalizer === "openai" || brandDetector === "openai" || featureMode === "openai" || featureEvidenceMode === "openai";

  function appendLog(event: ProgressEvent) {
    if (event.type === "run") {
      setRunLogs([
        {
          id: `run-${event.timestamp_ms}`,
          stage: "run",
          status: event.status || "started",
          message: event.message || "Run started",
          timestampMs: event.timestamp_ms,
        },
      ]);
      setActiveStage(event.message || "Run started");
      return;
    }
    if (event.type === "stage") {
      setRunLogs((current) => [
        ...current,
        {
          id: `${event.stage}-${event.status}-${event.timestamp_ms}`,
          stage: event.stage || "stage",
          status: event.status || "running",
          message: event.message || event.stage || "Stage update",
          timestampMs: event.timestamp_ms,
          durationMs: event.duration_ms,
        },
      ]);
      setActiveStage(event.message || event.stage || "Running");
      return;
    }
    if (event.type === "error") {
      setRunLogs((current) => [
        ...current,
        {
          id: `error-${event.timestamp_ms}`,
          stage: "error",
          status: "failed",
          message: event.error || "Run failed",
          timestampMs: event.timestamp_ms,
        },
      ]);
      setActiveStage("Run failed");
    }
  }

  async function consumeStream(response: Response) {
    if (!response.body) throw new Error("Streaming response body is not available.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalResult: ApiResult | null = null;
    let streamError = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line) as ProgressEvent;
        if (event.type === "result" && event.result) {
          finalResult = event.result;
          setRunLogs((current) => [
            ...current,
            {
              id: `result-${event.timestamp_ms}`,
              stage: "result",
              status: "completed",
              message: "Analysis finished",
              timestampMs: event.timestamp_ms,
            },
          ]);
          setActiveStage("Analysis finished");
        } else if (event.type === "error") {
          appendLog(event);
          streamError = event.error || "Analysis failed.";
        } else {
          appendLog(event);
        }
      }
    }

    if (buffer.trim()) {
      const event = JSON.parse(buffer) as ProgressEvent;
      if (event.type === "result" && event.result) finalResult = event.result;
      else if (event.type === "error") streamError = event.error || "Analysis failed.";
      else appendLog(event);
    }

    if (streamError) throw new Error(streamError);
    if (!finalResult?.ok) throw new Error(finalResult?.error || finalResult?.stderr || "Analysis failed.");
    return finalResult;
  }

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

  async function refreshRuns() {
    const response = await fetch("/api/runs");
    const data = (await response.json()) as { ok: boolean; runs: RunSummary[] };
    if (data.ok) setSavedRuns(data.runs);
  }

  function applyResultState(data: ApiResult, source: "sample" | "peec" | "csv" | "saved_run") {
    setResult(data);
    setResultSource(source);
    const hasStrictGaps = (data.overview || []).some((row) => Boolean(row.is_feature_visibility_gap));
    if (!hasStrictGaps) {
      setResultsMode("all");
    }
  }

  function goToResults() {
    if (result) {
      setWorkspaceView("results");
      setResultsTab("dashboard");
    }
  }

  function goToFeatures() {
    if (selectedRunId) {
      setWorkspaceView("results");
      setResultsTab("features");
    }
  }

  useEffect(() => {
    refreshRuns().catch(() => {});
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
    setRunLogs([]);
    setActiveStage("Running sample analysis");
    try {
      const response = await fetch("/api/analyze-sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_brand: "Peec AI", normalizer, brand_detector: brandDetector, feature_evidence_mode: featureEvidenceMode, embedding_backend: embeddingBackend, aggregation_mode: aggregationMode }),
      });
      const data = (await response.json()) as ApiResult;
      if (!response.ok || !data.ok) throw new Error(data.error || data.stderr || "Sample analysis failed.");
      applyResultState(data, "sample");
      setWorkspaceView("results");
      setResultsTab("dashboard");
      if (data.run_id) setSelectedRunId(data.run_id);
      await refreshRuns();
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
    setRunLogs([]);
    setActiveStage("Preparing analysis");
    try {
      if (!featureFile) throw new Error("Feature CSV or PDF is required.");

      const form = new FormData();
      let endpoint = "/api/analyze-peec-stream";
      if (dataSource === "csv") {
        if (!promptsCsv || !brandsCsv) throw new Error("CSV fallback requires Prompts CSV and Brands CSV.");
        form.append("prompts_csv", promptsCsv);
        form.append("brands_csv", brandsCsv);
        endpoint = "/api/analyze-stream";
      } else {
        form.append("project_id", projectId);
        form.append("start_date", startDate);
        form.append("end_date", endDate);
        form.append("limit", String(Math.max(1, Number.parseInt(peecLimit || "250", 10) || 250)));
        if (brandsCsv) form.append("brands_csv", brandsCsv);
      }
      form.append("feature_file", featureFile);
      form.append("target_brand", targetBrand);
      form.append("normalizer", normalizer);
      form.append("brand_detector", brandDetector);
      form.append("feature_evidence_mode", featureEvidenceMode);
      form.append("embedding_backend", embeddingBackend);
      form.append("aggregation_mode", aggregationMode);
      form.append("feature_mode", featureMode);

      const response = await fetch(endpoint, { method: "POST", body: form });
      if (!response.ok && !response.body) {
        const data = (await response.json()) as ApiResult;
        throw new Error(data.error || data.stderr || "Analysis failed.");
      }
      const data = await consumeStream(response);
      applyResultState(data, dataSource);
      setWorkspaceView("results");
      setResultsTab("dashboard");
      if (data.run_id) setSelectedRunId(data.run_id);
      await refreshRuns();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  async function openSavedRun(runId: string, targetTab: ResultsTab = "dashboard") {
    setLoading(true);
    setError("");
    setSelectedIndex(0);
    try {
      const response = await fetch(`/api/runs/${runId}`);
      const data = (await response.json()) as ApiResult;
      if (!response.ok || !data.ok) throw new Error(data.error || "Failed to load saved run.");
      applyResultState(data, "saved_run");
      setSelectedRunId(runId);
      setWorkspaceView("results");
      setResultsTab(targetTab);
      setRunLogs([]);
      setActiveStage(`Loaded saved run ${runId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load saved run.");
    } finally {
      setLoading(false);
    }
  }

  async function reaggregateRun(runIdOverride?: string) {
    const runId = runIdOverride || selectedRunId;
    if (!runId) return;
    setLoading(true);
    setError("");
    setRunLogs([]);
    setActiveStage("Re-aggregating saved run");
    try {
      const response = await fetch(`/api/runs/${runId}/reaggregate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ aggregation_mode: aggregationMode }),
      });
      const data = (await response.json()) as ApiResult;
      if (!response.ok || !data.ok) throw new Error(data.error || "Re-aggregation failed.");
      applyResultState(data, "saved_run");
      setWorkspaceView("results");
      setResultsTab("dashboard");
      if (data.run_id) setSelectedRunId(data.run_id);
      await refreshRuns();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Re-aggregation failed.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!selectedRunId) {
      setRunMappings([]);
      return;
    }
    let cancelled = false;
    setRunMappingsLoading(true);
    fetch(`/api/runs/${selectedRunId}/mappings?limit=1000`)
      .then((response) => response.json())
      .then((data: { ok: boolean; mappings: MappingRow[] }) => {
        if (!cancelled && data.ok) setRunMappings(data.mappings);
      })
      .catch(() => {
        if (!cancelled) setRunMappings([]);
      })
      .finally(() => {
        if (!cancelled) setRunMappingsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRunId]);

  return (
    <main className="app-shell app-shell-print">
      <section className="workspace workspace-print">
        <header className="print-header">
          <div className="print-brand">
            <div className="mark">
              <BarChart3 size={18} />
            </div>
            <strong>Feature Visibility</strong>
          </div>
          <nav className="print-steps">
            <button className={workspaceView === "setup" ? "active" : ""} onClick={() => setWorkspaceView("setup")}>
              <span>1</span>
              Setup
            </button>
            <button
              className={workspaceView === "results" ? "active" : ""}
              onClick={goToResults}
              disabled={!result}
            >
              <span>2</span>
              Results
            </button>
          </nav>
          <div className={`print-status ${apiOnline ? "online" : apiOnline === false ? "offline" : ""}`}>
            <span className="status-dot" />
            {apiOnline === null ? "Checking API" : apiOnline ? "API connected" : "API unavailable"}
          </div>
        </header>

        {error ? <div className="error-banner">{error}</div> : null}

        {workspaceView === "setup" ? (
          <section className="setup-screen">
            <div className="setup-hero">
              <h1>Find your AI visibility gaps</h1>
              <p>
                This tool checks whether AI assistants mention your product when users ask about specific
                features and shows where competitors are appearing instead.
              </p>
              <blockquote>
                One report that tells product, content, SEO, and engineering exactly which features are invisible
                to AI and who is winning in your place.
              </blockquote>
              <div className="hero-tags">
                <span>Product</span>
                <span>Content</span>
                <span>SEO</span>
                <span>Engineering</span>
              </div>
            </div>

            <div className="setup-stack">
              <section className="setup-card">
                <div className="setup-card-title">
                  <span className="step-dot">1</span>
                  <h2>Name your brand</h2>
                </div>
                <label className="field">
                  <span>Your brand name</span>
                  {brandOptions.length ? (
                    <select value={targetBrand} onChange={(event) => setTargetBrand(event.target.value)}>
                      {brandOptions.map((name) => <option key={name} value={name}>{name}</option>)}
                    </select>
                  ) : (
                    <input value={targetBrand} onChange={(event) => setTargetBrand(event.target.value)} />
                  )}
                </label>
                <p className="helper-copy">This is the brand we will look for inside AI-generated answers.</p>
              </section>

              <section className="setup-card">
                <div className="setup-card-title">
                  <span className="step-dot">2</span>
                  <h2>Upload your feature list</h2>
                </div>
                <label className="field">
                  <span>Feature list (CSV or PDF)</span>
                </label>
                <FilePicker label="Feature CSV or PDF" accept=".csv,.pdf" file={featureFile} onChange={setFeatureFile} />
                <p className="helper-copy">Each row should have a feature name and a short description of what it does.</p>
              </section>

              <details className="advanced-settings" open={dataSource === "csv"}>
                <summary>Advanced settings</summary>
                <div className="advanced-grid">
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
                      <FilePicker label="Brands CSV override" accept=".csv" file={brandsCsv} onChange={setBrandsCsv} />
                      <div className="notice">Prompts and responses come from Peec MCP. Brand CSV is optional.</div>
                    </div>
                  ) : (
                    <div className="compact-fields">
                      <FilePicker label="Prompts CSV" accept=".csv" file={promptsCsv} onChange={setPromptsCsv} />
                      <FilePicker label="Brands CSV" accept=".csv" file={brandsCsv} onChange={setBrandsCsv} />
                    </div>
                  )}

                  <div className="advanced-controls">
                    <SelectControl label="Prompt normalizer" value={normalizer} options={["openai_mock", "heuristic", "openai"]} onChange={(value) => setNormalizer(value as Mode)} />
                    <SelectControl label="Brand detector" value={brandDetector} options={["openai_mock", "keyword", "openai"]} onChange={(value) => setBrandDetector(value as BrandMode)} />
                    <SelectControl label="Feature evidence" value={featureEvidenceMode} options={["openai_mock", "keyword", "openai"]} onChange={(value) => setFeatureEvidenceMode(value as FeatureEvidenceMode)} />
                    <SelectControl label="PDF extraction" value={featureMode} options={["mock", "openai"]} onChange={setFeatureMode} />
                    <SelectControl label="Embeddings" value={embeddingBackend} options={["hash", "bge-m3"]} onChange={setEmbeddingBackend} />
                    <SelectControl label="Aggregation" value={aggregationMode} options={["response", "prompt", "prompt_model"]} onChange={setAggregationMode} />
                    <label className="field">
                      <span>Peec chat limit</span>
                      <input value={peecLimit} onChange={(event) => setPeecLimit(event.target.value.replace(/[^\d]/g, "") || "")} />
                    </label>
                  </div>
                  {realModeSelected ? <div className="notice">Real LLM modes require `OPENAI_API_KEY` in the API server environment.</div> : null}
                </div>
              </details>

              <div className="setup-actions">
                <button className="button button-primary button-hero" onClick={analyzeUpload} disabled={loading || !uploadReady || apiOnline === false}>
                  {loading ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
                  {loading ? activeStage || "Running analysis" : "Run analysis"}
                </button>
                <button className="button button-secondary" onClick={analyzeSample} disabled={loading || apiOnline === false}>
                  <FileText size={17} />
                  Load sample
                </button>
              </div>

              {!uploadReady && !result ? (
                <div className="setup-banner">
                  {dataSource === "peec"
                    ? "Choose a date range and upload one feature CSV or PDF. Project ID is optional."
                    : "Upload prompts, brands, and one feature CSV or PDF."}
                </div>
              ) : null}
            </div>
          </section>
        ) : null}

        {workspaceView === "results" ? (
          <>
        <section className="results-tabs">
          <div className="results-tabs-shell">
            <div className="results-tabs-copy">
              <h2>Analysis</h2>
              <p>{selectedRunSummary ? formatRunLabel(selectedRunSummary) : "Open a run and inspect it from multiple angles"}</p>
            </div>
            <div className="results-tabbar" role="tablist" aria-label="Analysis views">
              <button className={`results-tab ${resultsTab === "dashboard" ? "active" : ""}`} onClick={() => setResultsTab("dashboard")} aria-selected={resultsTab === "dashboard"}>
                <strong>Dashboard</strong>
                <span>Visibility summary</span>
              </button>
              <button className={`results-tab ${resultsTab === "history" ? "active" : ""}`} onClick={() => setResultsTab("history")} aria-selected={resultsTab === "history"}>
                <strong>History</strong>
                <span>Saved runs</span>
              </button>
              <button
                className={`results-tab ${resultsTab === "features" ? "active" : ""}`}
                onClick={() => setResultsTab("features")}
                disabled={!selectedRunId}
                aria-selected={resultsTab === "features"}
              >
                <strong>Features</strong>
                <span>Mapped queries</span>
              </button>
            </div>
          </div>
        </section>
        {resultsTab === "dashboard" ? (
          <>
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
                  <p>{selectedRunSummary ? formatRunLabel(selectedRunSummary) : `${visibleRows.length} visible rows`}</p>
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
                <div className="empty-state">
                  <Loader2 className="spin" size={28} />
                  Analyzing
                  <small>{activeStage || "Processing current run"}</small>
                </div>
              ) : visibleRows.length ? (
                <div className="gap-list">
                  {visibleRows.map((row, index) => (
                    <GapCard key={`${row.mapped_feature_name}-${row.cluster_label}-${index}`} row={row} selected={index === selectedIndex} onSelect={() => setSelectedIndex(index)} />
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  {result
                    ? resultsMode === "gaps" && sortedOverview.length
                      ? "This run has no strict gaps. Switch to All to inspect the saved results."
                      : "No rows match the current filter."
                    : "No selected run. Open a saved run from History or run a new analysis."}
                </div>
              )}
            </section>

            <section className="detail-panel" style={selected ? selectedDetailStyle : undefined}>
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
                    <div><span>Status</span><strong className={`traffic-${trafficTone(selected.target_visibility_status)}`}>{selected.target_visibility_status}</strong></div>
                    <div><span>Competitor</span><strong>{selected.top_competitor_brand_name || "-"}</strong></div>
                  </div>
                  <div className="detail-metrics">
                    <div><span>Feature evidence</span><strong>{selected.target_feature_visible_count ?? 0}/{selected.prompt_count}</strong></div>
                    <div><span>Prompts</span><strong>{selected.prompt_count}</strong></div>
                    <div><span>Signal</span><strong className={`traffic-${trafficTone(selected.signal || selected.target_visibility_status)}`}>{selected.signal || "-"}</strong></div>
                  </div>
                  <div className="reason-box">{selected.gap_reason}</div>
                  <BreakdownChart title="Visibility comparison" items={selectedComparison} />
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
                  <BreakdownChart title="Models" items={selectedModelBreakdown} suffix="" />
                  <BreakdownChart title="Sources" items={selectedSourceBreakdown} suffix="" />
                </div>
              ) : (
                <div className="empty-state">Select a gap to inspect.</div>
              )}
            </section>
          </div>
          </>
        ) : null}

        {resultsTab === "history" ? (
          <section className="history-page">
            <div className="section-heading">
              <div>
                <h2>History</h2>
                <p>{savedRuns.length} saved runs</p>
              </div>
            </div>
            {savedRuns.length ? (
              <div className="history-list">
                {savedRuns.map((run) => (
                  <button
                    key={run.run_id}
                    className={`history-card ${run.run_id === selectedRunId ? "selected" : ""}`}
                    onClick={() => openSavedRun(run.run_id, "dashboard")}
                    disabled={loading}
                  >
                    <div className="history-card-top">
                      <div>
                        <strong>{formatRunLabel(run)}</strong>
                        <p>{sourceLabel(run.source)} · {run.prompt_rows} rows · {run.overview_rows} overview rows</p>
                      </div>
                      {run.run_id === selectedRunId ? <span className="strict-pill">selected</span> : null}
                    </div>
                    <div className="history-actions">
                      <span className="history-inline-hint">Click anywhere to load this run</span>
                      <button
                        className="button button-secondary"
                        onClick={(event) => {
                          event.stopPropagation();
                          reaggregateRun(run.run_id);
                        }}
                        disabled={loading}
                      >
                        Re-aggregate
                      </button>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <div className="empty-state">No saved runs yet. Run an analysis to create one.</div>
            )}
          </section>
        ) : null}

        {resultsTab === "features" ? (
          <section className="mappings-page">
            <div className="section-heading">
              <div>
                <h2>Features</h2>
                <p>{selectedRunSummary ? `${formatRunLabel(selectedRunSummary)} · queries mapped to each feature` : "Queries mapped to each feature for this run"}</p>
              </div>
              <div className="toolbar">
                <CsvDownload rows={runMappings as unknown as Record<string, unknown>[]} filename="saved_run_feature_mappings.csv" />
              </div>
            </div>
            {!selectedRunId ? (
              <div className="empty-state">No selected run. Go to History and open a run first.</div>
            ) : runMappingsLoading ? (
              <div className="empty-state">Loading feature mappings...</div>
            ) : groupedRunMappings.length ? (
              <div className="feature-mapping-groups">
                {groupedRunMappings.map((group) => (
                  <div
                    key={group.featureName}
                    className="feature-mapping-card"
                    style={featureAccentStyle(group.queries[0]?.mapped_feature_id || group.featureName)}
                  >
                    <div className="feature-mapping-header">
                      <strong>{group.featureName}</strong>
                      <span>{group.queries.length} queries</span>
                    </div>
                    <ul className="query-list">
                      {group.queries.map((row) => (
                        <li key={`${group.featureName}-${row.prompt_id}-${row.canonical_query}`}>
                          <strong>{row.canonical_query || row.original_prompt || row.prompt_id}</strong>
                          <br />
                          {row.original_prompt || row.prompt_id}
                          <br />
                          <span className="mapping-meta">
                            {(row.engine || "unknown model")} · feature {row.feature_present ? "detected" : "missing"} · brand {row.brand_present ? "present" : "absent"}
                            {row.source_domains ? ` · ${row.source_domains}` : ""}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state">No feature-query mappings were saved for this run.</div>
            )}
          </section>
        ) : null}
          </>
        ) : null}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
