# Minimal Operational System Checklist

Goal: move from a one-off analysis tool to a small reusable system with CSV-backed persistence.

## Scope

- [x] Save every run with a stable `run_id`
- [x] Save a run index that can be listed later
- [x] Save the feature set used for each run
- [x] Save the brands set used for each run
- [x] Save row-level query mappings for each run
- [x] Save coverage, gap overview, gap details, and summary for each run
- [x] Expose saved runs through API
- [x] Expose saved row-level mappings through API
- [x] Support post-run re-aggregation from saved mappings
- [x] Show saved runs in the UI
- [x] Show queries/prompts that belong to a selected feature cluster in the UI
- [x] Reuse cached Peec/OpenAI work for demo reruns

## Acceptance Checks

- [x] A new run creates a persistent run folder on disk
- [x] The run folder contains feature, brand, and mapping snapshots
- [x] The API can list previous runs after restart
- [x] The API can load a previous run without re-running Peec or OpenAI
- [x] The API can re-aggregate a saved run with `response`, `prompt`, or `prompt_model`
- [x] The UI can open a saved run and inspect its saved outputs
- [x] The UI can show prompt/query mappings for the selected feature cluster
- [x] Tests cover persistence and re-aggregation behavior
