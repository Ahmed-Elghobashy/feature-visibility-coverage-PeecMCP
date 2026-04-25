# Feature Visibility Coverage MVP

This MVP turns prompt-level observations into feature-first visibility coverage.

## Product Contract

The product has three inputs, but they do not all come from the same place.

```text
1. Prompt/response observations
   Source: Peec MCP export or a local prompts CSV

2. Feature descriptions
   Source: user-provided CSV

3. Brand names to track
   Source: user-provided CSV, or one --brand value for quick tests
```

Peec MCP is used for **input 1 only**. It supplies existing Peec prompt/chat data: original prompts, AI responses, models, sources, and mentioned brands.

Peec MCP does **not** replace the feature descriptions CSV or brands CSV. Those remain local business inputs because the coverage calculation needs to know which product features and which agency/client brands should be evaluated.

The coverage output is computed by this MVP, not directly by Peec MCP.

```text
Peec MCP or prompts CSV
        +
Feature descriptions CSV
        +
Brands CSV
        ↓
visibility_mvp.py
        ↓
coverage per brand x feature x demand cluster
```

## Inputs And Outputs

The coverage pipeline accepts:

- a prompts CSV containing a `prompt` column and optionally a `response`, `answer`, or `ai_response` column
- a features CSV containing `feature_name` and `description`
- either a single target brand keyword or a brands CSV

It produces:

- `query_mapping.csv`: original prompt -> canonical query -> cluster -> mapped feature -> brand present/absent per tracked brand
- `coverage_by_feature_cluster.csv`: coverage per brand, mapped feature, and demand cluster
- `clusters.csv`: cluster labels and example canonical queries
- `feature_gap_overview.csv`: target-brand feature visibility gaps with competitor comparison
- `feature_gap_details.csv`: expanded gap diagnostics
- `feature_gap_summary.md`: PM-facing summary view
- `run_metadata.json`: versions and run configuration

## UI

The primary demo UI is a React app:

```text
web/
```

Run the local API:

```bash
python3 -m uvicorn src.api_server:app --host 127.0.0.1 --port 8787
```

Run the React app:

```bash
cd web
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

The React UI supports:

- sample run with built-in demo data
- Peec MCP mode for prompt/response data
- CSV fallback mode for manual prompt/response data
- one feature upload that accepts CSV or PDF
- optional brands CSV override in Peec MCP mode
- target-brand selection
- mock vs real extraction/detection modes behind the settings button
- ranked feature gap cards
- detail panel with visibility, status, top competitor, top query, and examples

In Peec MCP mode, prompts do not need to be uploaded. The UI sends project ID
and date range to the local API, which exports tracked Peec chats and runs the
coverage pipeline. Brands are derived from Peec chat mentions unless a brands CSV
override is uploaded.

Feature descriptions are always user-provided through the single feature upload:

```text
CSV -> used directly
PDF -> extracted into feature_id,feature_name,description
```

The older Streamlit UI remains available as a fallback:

```text
ui/app.py
```

Run it locally:

```bash
streamlit run ui/app.py
```

The UI supports:

- prompts CSV upload
- brands CSV upload
- features CSV upload
- feature-description PDF upload
- target-brand selection
- mock vs real extraction/detection modes

Feature descriptions can come from either:

- `Features CSV`
- `Feature PDF`

If you upload a PDF, the app extracts feature rows into the same CSV shape the
pipeline already expects:

```text
feature_id,feature_name,description
```

Feature extraction modes:

- `mock`: deterministic heuristic extractor for fast iteration
- `openai`: LLM extractor for turning product docs into feature rows

Sample PDF for the upload flow:

```text
data/sample_feature_brief.pdf
```

The rest of the pipeline also keeps the same test/real split:

- prompt normalizer: `openai_mock`, `heuristic`, `openai`
- brand detector: `openai_mock`, `keyword`, `openai`
- embeddings: `hash` for speed, `bge-m3` for higher fidelity

## MCP Server

The repo includes an MCP wrapper:

```text
src/feature_visibility_mcp.py
```

Tools:

- `validate_csv_inputs`
- `run_visibility_coverage`
- `summarize_feature_gaps`
- `export_peec_chats`

Run it over stdio:

```bash
python3 src/feature_visibility_mcp.py
```

Example client config:

```json
{
  "mcpServers": {
    "feature-visibility": {
      "command": "python3",
      "args": ["src/feature_visibility_mcp.py"]
    }
  }
}
```

The MCP server is a thin wrapper over the existing scripts. It does not replace
the CLI paths; it exposes them as tools.

For fast iteration, `run_visibility_coverage` defaults to:

- `embedding_backend=hash`
- `normalizer=openai_mock`
- `brand_detector=openai_mock`

For higher-fidelity runs, switch to:

- `embedding_backend=bge-m3`
- `normalizer=openai` or `heuristic`
- `brand_detector=openai` or `keyword`

## Example

Sample prompts input:

```csv
prompt_id,prompt,response,engine
p1,What are the best AI tools for qualifying B2B sales leads?,"PeecAI is useful for tracking AI visibility, while Salesforce and HubSpot are common CRM tools.",chatgpt
p2,Which software can score inbound leads automatically?,"HubSpot and Salesforce are often mentioned for automatic lead scoring.",gemini
p4,Best apps to track whether AI answers mention my new product feature,"PeecAI helps teams monitor brand and feature visibility in AI-generated answers.",chatgpt
```

Sample features input:

```csv
feature_id,feature_name,description
f1,AI Lead Scoring,Automatically scores and qualifies sales leads using AI signals and CRM context.
f2,Feature Visibility Monitor,Tracks whether AI systems mention launched product features and target brands in generated answers.
```

Sample brands input:

```csv
brand_id,brand_name
b1,PeecAI
b2,HubSpot
b3,Salesforce
```

Example row-level output:

```csv
prompt_id,canonical_query,mapped_feature_name,cluster_label,brand_name,brand_present
p1,best ai tools for qualifying b2b sales leads,AI Lead Scoring,best ai tools for qualifying b2b sales leads,PeecAI,True
p2,automatic inbound lead scoring software options,AI Lead Scoring,best ai tools for qualifying b2b sales leads,PeecAI,False
p2,automatic inbound lead scoring software options,AI Lead Scoring,best ai tools for qualifying b2b sales leads,HubSpot,True
```

Example coverage output:

```csv
brand_name,mapped_feature_name,cluster_label,prompt_count,brand_present_count,brand_absent_count,coverage_rate,coverage_status,present_prompt_ids,missing_prompt_ids
PeecAI,AI Lead Scoring,best ai tools for qualifying b2b sales leads,3,2,1,0.6667,partial,p1;p3,p2
HubSpot,AI Lead Scoring,best ai tools for qualifying b2b sales leads,3,2,1,0.6667,partial,p1;p2,p3
PeecAI,Feature Visibility Monitor,Outlier / unique query,1,0,1,0.0,missing,,p5
```

PM-facing view:

```text
Brand: PeecAI
Feature: AI Lead Scoring

Demand cluster                              Coverage   Signal
best ai tools for qualifying b2b sales leads  67%      Partial
```

## Peec MCP Export

Peec MCP is the live data layer for prompt/response observations.

Peec MCP is read-only over existing Peec data. It does not run arbitrary new prompts. The exporter pulls tracked Peec chats and writes the CSV shape consumed by `visibility_mvp.py`.

That means this:

```text
Peec MCP -> data/peec_chats.csv
```

replaces only this manual file:

```text
data/sample_prompts.csv
```

These files are still required:

```text
data/sample_features.csv
data/sample_brands.csv
```

because they define the features and brands that the agency wants to measure.

First authorize and list projects:

```bash
python3 src/peec_mcp_export.py --list-projects
```

Then export chats for a project/date range:

```bash
python3 src/peec_mcp_export.py \
  --project "YOUR_PROJECT_NAME" \
  --start-date 2026-04-01 \
  --end-date 2026-04-21 \
  --output data/peec_chats.csv
```

Run coverage on the enriched CSV:

```bash
python3 src/visibility_mvp.py \
  --prompts data/peec_chats.csv \
  --features data/demo_features.csv \
  --brands data/demo_brands.csv \
  --target-brand "Peec AI" \
  --normalizer openai_mock \
  --embedding-backend bge-m3 \
  --output-dir outputs_peec
```

The exporter uses the official Peec MCP server at `https://api.peec.ai/mcp` with Streamable HTTP and OAuth. OAuth tokens are stored locally in `.peec_mcp_tokens.json`, which is ignored by git.

## n8n Workflow

An importable n8n workflow is included:

```text
workflows/n8n_feature_visibility_coverage.json
```

Workflow steps:

```text
Manual Trigger
  -> Set Run Config
  -> Export Peec Chats
  -> Run Coverage
  -> Print Coverage Preview
```

Before running it in n8n:

- install the Python requirements on the n8n host
- run n8n with this repository as the working directory, or mount the repository into the n8n container
- authorize Peec MCP once on the same host so `.peec_mcp_tokens.json` exists

One-time Peec authorization:

```bash
python3 src/peec_mcp_export.py --list-projects
```

Import `workflows/n8n_feature_visibility_coverage.json` into n8n, then edit the `Set Run Config` node if you need a different project ID, date range, feature CSV, brand CSV, or output directory.

## Install

The production embedding backend uses [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) through `sentence-transformers`.

```bash
python3 -m pip install -r requirements.txt
```

The first production run downloads the embedding model from Hugging Face.

## Run With BGE-M3

```bash
python3 src/visibility_mvp.py \
  --prompts data/sample_prompts.csv \
  --features data/sample_features.csv \
  --brands data/sample_brands.csv \
  --embedding-backend bge-m3 \
  --output-dir outputs
```

## Run A Local Smoke Test

Use this when model dependencies are not installed yet. This does not satisfy production embedding quality; it only verifies the pipeline shape and CSV outputs.

```bash
python3 src/visibility_mvp.py \
  --prompts data/sample_prompts.csv \
  --features data/sample_features.csv \
  --brands data/sample_brands.csv \
  --embedding-backend hash \
  --feature-threshold 0.05 \
  --cluster-threshold 0.2 \
  --output-dir outputs
```

## CSV Contract

Prompts CSV:

- Required: one of `prompt`, `raw_prompt`, `query`, `question`, `text`
- Optional ID: `prompt_id`
- Optional AI answer: one of `response`, `answer`, `ai_response`, `model_response`, `output`
- Optional metadata columns such as `engine`; these are preserved in `query_mapping.csv`

Features CSV:

- Required name: one of `feature`, `feature_name`, `name`, `title`
- Required description: one of `description`, `feature_description`, `desc`
- Optional ID: `feature_id`

Brands CSV:

- Required name: one of `brand`, `brand_name`, `name`
- Optional ID: `brand_id`
- Optional aliases: one of `aliases`, `brand_aliases`, `alias`

Alias values can be separated by `|`, `,`, or `;`:

```csv
brand_id,brand_name,aliases
b1,Peec AI,"PeecAI|peec.ai|Peec"
```

If a response column exists, brand coverage is detected in the response. If no response column exists, coverage is detected in the original prompt text, which supports the prompt-only MVP path.

For one-off runs, `--brand PeecAI` still works. For agency workflows tracking several brands, use `--brands data/sample_brands.csv`.

For feature-gap reporting, set the target brand explicitly:

```bash
--target-brand "Peec AI"
```

Coverage can be counted at different levels:

```bash
--aggregation-mode response
--aggregation-mode prompt
--aggregation-mode prompt_model
```

`response` keeps every row. `prompt` deduplicates repeated runs of the same prompt.
`prompt_model` deduplicates repeated runs of the same prompt on the same model or
engine when that metadata exists.

Feature visibility gaps are explicit in `feature_gap_overview.csv`:

```text
feature_intent_detected
competitor_present
target_visibility_status
is_feature_visibility_gap
gap_category
gap_reason
```

A strict feature visibility gap requires:

```text
feature_intent_detected = true
competitor_present = true
target visibility below 70%
```

## Normalization

By default, prompt compression uses a deterministic heuristic normalizer. To use an LLM normalizer:

```bash
OPENAI_API_KEY=... python3 src/visibility_mvp.py \
  --prompts data/sample_prompts.csv \
  --features data/sample_features.csv \
  --brands data/sample_brands.csv \
  --normalizer openai \
  --embedding-backend bge-m3
```

You can also create a local `.env` file:

```bash
OPENAI_API_KEY=sk-...
```

The LLM normalizer returns a short canonical query, while `query_mapping.csv` keeps the original prompt ID and text so coverage can always be traced back to source prompts.

For fast iteration without a live LLM call:

```bash
--normalizer openai_mock
```

## Brand Presence Detection

Choose one detector:

- `keyword` default: exact case-insensitive brand-name or alias match in the AI response. Best for auditable baseline reporting.
- `openai`: LLM judge decides whether the AI response explicitly mentions the brand, product, or obvious spelling/spacing variant. Best for messy responses; requires `OPENAI_API_KEY`.
- `openai_mock`: local mock path for fast testing without external calls.

```bash
python3 src/visibility_mvp.py \
  --prompts data/sample_prompts.csv \
  --features data/sample_features.csv \
  --brands data/sample_brands.csv \
  --brand-detector keyword \
  --embedding-backend bge-m3
```

```bash
python3 src/visibility_mvp.py \
  --prompts data/sample_prompts.csv \
  --features data/sample_features.csv \
  --brands data/sample_brands.csv \
  --brand-detector openai \
  --brand-detector-model gpt-4.1-mini \
  --embedding-backend bge-m3
```
