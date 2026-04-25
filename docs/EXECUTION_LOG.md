# Execution Log

This document tracks the implementation path from the current MVP to v0.5 and v1.0.

## Current Goal

Move from a coverage-oriented MVP to a product-oriented system for identifying feature visibility gaps in AI answers.

## Iteration Plan

### v0.5

Product layer on top of the current pipeline:

- explicit target brand selection
- competitor comparison
- visibility share
- consistency band
- gap severity
- feature gap overview output
- feature gap detail output
- PM-facing summary artifact

### v1.0

Packaging and tool surface:

- MCP server for export, coverage, and summary tools
- mock/test modes where external model calls would otherwise slow iteration
- improved README and operating docs

## Notes

- Peec MCP remains the raw data source.
- Feature and brand CSVs remain product-specific inputs.
- Generated Peec data and OAuth tokens stay ignored by git.
