#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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

    if not run_clicked:
        return

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

        st.subheader("Run Log")
        st.code(stdout or "(no stdout)", language="text")
        if stderr:
            st.code(stderr, language="text")
        if not ok:
            st.error("Pipeline run failed.")
            return

        overview = load_table(output_dir / "feature_gap_overview.csv")
        details = load_table(output_dir / "feature_gap_details.csv")
        summary_md = (output_dir / "feature_gap_summary.md").read_text(encoding="utf-8")
        metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))

        if feature_source == "Feature PDF":
            with st.expander("Extracted Features", expanded=True):
                st.dataframe(features_df, use_container_width=True)
            with st.expander("Extracted PDF Text Preview"):
                st.text_area("PDF text", extracted_text[:12000], height=320)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Feature Gap Overview")
            st.dataframe(overview, use_container_width=True)
        with col2:
            st.subheader("Run Metadata")
            st.json(metadata)

        st.subheader("PM Summary")
        st.markdown(summary_md)

        with st.expander("Feature Gap Details"):
            st.dataframe(details, use_container_width=True)

        st.download_button(
            "Download feature_gap_overview.csv",
            data=(output_dir / "feature_gap_overview.csv").read_bytes(),
            file_name="feature_gap_overview.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download extracted features CSV",
            data=features_df.to_csv(index=False).encode("utf-8"),
            file_name="features.csv",
            mime="text/csv",
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
