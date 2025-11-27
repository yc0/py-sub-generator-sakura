# Combined Project Summary

This document consolidates recent cleanup, optimization, progress and summary materials into a single reference. It is intended as a concise, de-duplicated index for the most important information in `docs/`.

## High-level highlights

- Cross-platform hardware acceleration (Apple VideoToolbox, NVIDIA CUDA) with robust fallback.
- Native Whisper ASR integration for stable timestamped transcription.
- SakuraLLM integration with prompt-engineering support (prompt templates in `prompts/`).
- VAD improvements and UI controls for runtime tuning.
- Extensive test-suite and performance benchmarks.

## Where to read deeper

- Hardware acceleration and benchmarks: `COMPLETE_PROGRESS_SUMMARY.md`
- Cleanup & conversion decisions: `CLEANUP_SUMMARY.md`, `CLEANUP_RESULTS.md`
- Progress log and recent feature work: `PROGRESS_SUMMARY.md`
- Sakura-specific integration and guidance: `SAKURA_GUIDE.md`, `SAKURA_UI_INTEGRATION.md`
- Setup and Apple Silicon guide: `SETUP_GUIDE.md`, `APPLE_SILICON_GUIDE.md`

## Consolidated action items (current)

1. Curate and expand prompt templates in `prompts/` and ensure sensitive/unsafe templates are audited.
2. Run E2E tests across representative videos (long/short/noisy) to validate VAD + ASR + translation behavior.
3. Add unit tests for prompt-template loading and for `SakuraTranslator.create_from_config()` behavior.
4. Consider merging duplicates into this file when longer-term maintenance is desired; for now this file acts as a canonical index.

## Rationale for consolidation

Multiple markdown files in `docs/` contained overlapping information (cleanup results, optimization summaries, and progress summaries). This file serves as a single-entry reference and points to the authoritative files for deeper details.

---

_Generated automatically by a repo maintenance task on Nov 27, 2025._
