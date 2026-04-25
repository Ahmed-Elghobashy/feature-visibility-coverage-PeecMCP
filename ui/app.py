#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feature_extraction import extract_features_from_pdf  # noqa: E402


VISIBILITY_SCRIPT = ROOT / "src" / "visibility_mvp.py"


st.set_page_config(
    page_title="Feature Visibility Gaps",
    layout="wide",
)


def save_upload(upload, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(upload.getvalue())
    return path


def run_pipeline(
    prompts_csv: Path,
    features_csv: Path,
    brands_csv: Path,
    target_brand: str,
    output_dir: Path,
    normalizer: str,
    brand_detector: str,
    embedding_backend: str,
) -> tuple[bool, str, str]:
    args = [
        sys.executable,
        str(VISIBILITY_SCRIPT),
        "--prompts",
        str(prompts_csv),
        "--features",
        str(features_csv),
        "--brands",
        str(brands_csv),
        "--target-brand",
        target_brand,
        "--normalizer",
        normalizer,
        "--brand-detector",
        brand_detector,
        "--embedding-backend",
        embedding_backend,
        "--output-dir",
        str(output_dir),
    ]
    if embedding_backend == "hash":
        args.extend(["--feature-threshold", "0.05", "--cluster-threshold", "0.2"])
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0, result.stdout, result.stderr


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def as_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def severity_rank(value: str) -> int:
    order = {"high": 0, "medium": 1, "low": 2}
    return order.get(str(value).casefold(), 3)


def render_share_bar(value: Any) -> str:
    try:
        share = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        share = 0.0
    percent = int(round(share * 100))
    return (
        f"<div style='height:10px;background:#e5e7eb;border-radius:6px;overflow:hidden;'>"
        f"<div style='width:{percent}%;height:10px;background:#111827;'></div>"
        f"</div>"
    )


def render_overview_cards(overview: pd.DataFrame) -> None:
    if overview.empty:
        st.info("No feature gaps were generated.")
        return

    working = overview.copy()
    working["_severity_rank"] = working["gap_severity"].map(severity_rank)
    working = working.sort_values(
        ["_severity_rank", "visibility_share", "mapped_feature_name"],
        ascending=[True, True, True],
    )

    st.subheader("Ranked Feature Gaps")
    for _, row in working.iterrows():
        with st.container(border=True):
            top = st.columns([2.2, 1.2, 1.4, 1.2])
            top[0].markdown(f"**{row['mapped_feature_name']}**  \n{row['cluster_label']}")
            top[1].metric("Visibility share", as_percent(row["visibility_share"]))
            top[2].markdown(f"**Signal**  \n{row['signal']}")
            top[3].markdown(f"**Top competitor**  \n{row['top_competitor_brand_name'] or '-'}")
            st.markdown(render_share_bar(row["visibility_share"]), unsafe_allow_html=True)
            st.caption(
                f"Category: {row.get('gap_category', '-')} | "
                f"Strict gap: {bool(row.get('is_feature_visibility_gap', False))} | "
                f"Consistency: {row['consistency_band']} | "
                f"Gap severity: {row['gap_severity']} | "
                f"Prompt count: {int(row['prompt_count'])}"
            )


def render_feature_detail(overview: pd.DataFrame, details: pd.DataFrame) -> None:
    if overview.empty:
        return

    feature_names = overview["mapped_feature_name"].dropna().astype(str).drop_duplicates().tolist()
    selected_feature = st.selectbox("Feature detail", options=feature_names, index=0)
    feature_overview = overview[overview["mapped_feature_name"] == selected_feature].copy()
    feature_overview["_severity_rank"] = feature_overview["gap_severity"].map(severity_rank)
    feature_overview = feature_overview.sort_values(
        ["_severity_rank", "visibility_share", "cluster_label"],
        ascending=[True, True, True],
    )

    for _, row in feature_overview.iterrows():
        with st.container(border=True):
            st.markdown(f"**Demand cluster**  \n{row['cluster_label']}")
            cols = st.columns(4)
            cols[0].metric("Visibility share", as_percent(row["visibility_share"]))
            cols[1].metric("Prompt count", int(row["prompt_count"]))
            cols[2].markdown(f"**Signal**  \n{row['signal']}")
            cols[3].markdown(f"**Top competitor**  \n{row['top_competitor_brand_name'] or '-'}")

            st.markdown(render_share_bar(row["visibility_share"]), unsafe_allow_html=True)
            st.caption(f"Category: {row.get('gap_category', '-')} | Reason: {row.get('gap_reason', '-')}")
            st.caption(f"Top query: {row['top_query']}")

            detail_rows = details[
                (details["mapped_feature_id"] == row["mapped_feature_id"])
                & (details["cluster_id"] == row["cluster_id"])
            ]
            if not detail_rows.empty:
                detail = detail_rows.iloc[0]
                example_queries = json.loads(detail.get("example_queries_json", "[]") or "[]")
                brand_comparison = json.loads(detail.get("brand_comparison_json", "[]") or "[]")

                if example_queries:
                    st.markdown("**Example queries**")
                    for query in example_queries[:5]:
                        st.markdown(f"- {query}")

                if brand_comparison:
                    comparison_df = pd.DataFrame.from_records(brand_comparison)
                    if "coverage_rate" in comparison_df.columns:
                        comparison_df["coverage_rate"] = comparison_df["coverage_rate"].map(as_percent)
                    st.markdown("**Competitor comparison**")
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)


def render_results(result: dict[str, Any]) -> None:
    overview = result["overview"]
    details = result["details"]
    summary_md = result["summary_md"]
    metadata = result["metadata"]
    features_df = result["features_df"]
    extracted_text = result["extracted_text"]
    feature_source = result["feature_source"]

    log_tab, overview_tab, detail_tab, raw_tab = st.tabs(
        ["Run Log", "Overview", "Detail View", "Raw Outputs"]
    )

    if not overview.empty:
        strict_gap_count = int(overview.get("is_feature_visibility_gap", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        avg_visibility = overview["visibility_share"].astype(float).mean()
        feature_count = int(overview["mapped_feature_name"].nunique())
        cluster_count = int(overview["cluster_id"].nunique())
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Strict gaps", strict_gap_count)
        kpi2.metric("Average visibility", as_percent(avg_visibility))
        kpi3.metric("Features covered", feature_count)
        kpi4.metric("Demand clusters", cluster_count)

    with log_tab:
        st.code(result["stdout"] or "(no stdout)", language="text")
        if result["stderr"]:
            st.code(result["stderr"], language="text")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(summary_md)
        with col2:
            st.json(metadata)

        if feature_source == "Feature PDF":
            with st.expander("Extracted Features", expanded=True):
                st.dataframe(features_df, use_container_width=True, hide_index=True)
            with st.expander("Extracted PDF Text Preview"):
                st.text_area("PDF text", extracted_text[:12000], height=320)

    with overview_tab:
        render_overview_cards(overview)
        st.download_button(
            "Download feature_gap_overview.csv",
            data=result["overview_csv"],
            file_name="feature_gap_overview.csv",
            mime="text/csv",
        )

    with detail_tab:
        render_feature_detail(overview, details)

    with raw_tab:
        st.subheader("Feature Gap Overview")
        st.dataframe(overview, use_container_width=True, hide_index=True)
        st.subheader("Feature Gap Details")
        st.dataframe(details, use_container_width=True, hide_index=True)
        st.download_button(
            "Download features.csv",
            data=result["features_csv"],
            file_name="features.csv",
            mime="text/csv",
        )


def main() -> None:
    st.title("Feature Visibility Gaps in AI Answers")
    st.caption("Upload prompts, brands, and feature definitions. Optionally upload a PDF and extract features into CSV-ready rows.")

    with st.sidebar:
        st.subheader("Run Modes")
        feature_mode = st.selectbox(
            "Feature extraction mode",
            options=["mock", "openai"],
            index=0,
            help="mock is deterministic and fast. openai uses an LLM to turn the PDF into feature rows.",
        )
        normalizer = st.selectbox(
            "Prompt normalizer",
            options=["openai_mock", "heuristic", "openai"],
            index=0,
        )
        brand_detector = st.selectbox(
            "Brand detector",
            options=["openai_mock", "keyword", "openai"],
            index=0,
        )
        embedding_backend = st.selectbox(
            "Embedding backend",
            options=["hash", "bge-m3"],
            index=0,
            help="hash is for fast UI iteration. bge-m3 is the production path.",
        )

    prompts_upload = st.file_uploader("Prompts CSV", type=["csv"], key="prompts")
    brands_upload = st.file_uploader("Brands CSV", type=["csv"], key="brands")

    feature_source = st.radio(
        "Feature input source",
        options=["Features CSV", "Feature PDF"],
        horizontal=True,
    )
    feature_csv_upload = None
    feature_pdf_upload = None
    if feature_source == "Features CSV":
        feature_csv_upload = st.file_uploader("Features CSV", type=["csv"], key="features_csv")
    else:
        feature_pdf_upload = st.file_uploader("Feature Description PDF", type=["pdf"], key="features_pdf")

    target_brand = None
    if brands_upload is not None:
        brands_df = pd.read_csv(brands_upload)
        if "brand_name" in brands_df.columns and not brands_df.empty:
            target_brand = st.selectbox(
                "Target brand",
                options=brands_df["brand_name"].dropna().astype(str).tolist(),
                index=0,
            )

    run_clicked = st.button("Run visibility analysis", type="primary", use_container_width=True)

    if run_clicked:
        if prompts_upload is None or brands_upload is None:
            st.error("Prompts CSV and Brands CSV are required.")
            return
        if feature_source == "Features CSV" and feature_csv_upload is None:
            st.error("Upload a Features CSV or switch to Feature PDF.")
            return
        if feature_source == "Feature PDF" and feature_pdf_upload is None:
            st.error("Upload a feature description PDF.")
            return
        if not target_brand:
            st.error("Brands CSV must include at least one brand_name.")
            return

        workdir = Path(tempfile.mkdtemp(prefix="feature_visibility_ui_"))
        try:
            prompts_path = save_upload(prompts_upload, workdir / "inputs" / "prompts.csv")
            brands_path = save_upload(brands_upload, workdir / "inputs" / "brands.csv")

            extracted_text = ""
            if feature_source == "Features CSV":
                features_path = save_upload(feature_csv_upload, workdir / "inputs" / "features.csv")
                features_df = pd.read_csv(features_path)
            else:
                features_df, extracted_text = extract_features_from_pdf(
                    feature_pdf_upload.getvalue(),
                    mode=feature_mode,
                )
                if features_df.empty:
                    st.error("No usable features were extracted from the PDF.")
                    if extracted_text:
                        st.text_area("Extracted PDF text preview", extracted_text[:6000], height=240)
                    return
                features_path = workdir / "inputs" / "features_from_pdf.csv"
                features_df.to_csv(features_path, index=False)

            output_dir = workdir / "outputs"
            ok, stdout, stderr = run_pipeline(
                prompts_csv=prompts_path,
                features_csv=features_path,
                brands_csv=brands_path,
                target_brand=target_brand,
                output_dir=output_dir,
                normalizer=normalizer,
                brand_detector=brand_detector,
                embedding_backend=embedding_backend,
            )

            if not ok:
                st.error("Pipeline run failed.")
                st.code(stdout or "(no stdout)", language="text")
                if stderr:
                    st.code(stderr, language="text")
                return

            st.session_state["last_result"] = {
                "overview": load_table(output_dir / "feature_gap_overview.csv"),
                "details": load_table(output_dir / "feature_gap_details.csv"),
                "summary_md": (output_dir / "feature_gap_summary.md").read_text(encoding="utf-8"),
                "metadata": json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8")),
                "features_df": features_df,
                "extracted_text": extracted_text,
                "feature_source": feature_source,
                "stdout": stdout,
                "stderr": stderr,
                "overview_csv": (output_dir / "feature_gap_overview.csv").read_bytes(),
                "features_csv": features_df.to_csv(index=False).encode("utf-8"),
            }
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    result = st.session_state.get("last_result")
    if result:
        render_results(result)


if __name__ == "__main__":
    main()
