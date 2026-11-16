# High-Signal Reading Guides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain the relevant Lexsi tools and research in the user's approved reading order without unexplained catalogs or obsolete project assumptions.

**Architecture:** Rewrite the guides in place using the existing source review, not a new repository survey. Each tool guide explains its problem, mechanism, one connected Aya example, selection criteria and consequential limitations. Preserve `13` and `15` byte-for-byte; retain the checked BibTeX in `14`.

**Tech Stack:** Markdown, Python standard-library checks, existing source-document backup.

---

## Scope and execution

The user approved direct execution and requested speed and compact guides. Work inline; no delegation, GPU execution, package installation or unrelated cleanup. This directory is not a Git repository, so no worktree or commit step applies.

Reading order: **01 → 02 → 03 → 04 → 05 → 07 → 11 → 12 → 13 → 14**.

Backup: `C:/Users/AstralK/AppData/Local/Temp/opencode/high-signal-docs-6336ceca7f044e61afdd2f7967f5e7c4`.

### Task 1: Check the documentation contract

- [x] Add `C:/Users/AstralK/AppData/Local/Temp/opencode/check_high_signal_docs.py` to check links, fences, Python syntax, required concepts, navigation, retired framing and protected records.
- [x] Run `python C:/Users/AstralK/AppData/Local/Temp/opencode/check_high_signal_docs.py`; expect the existing guides to fail the new structure and reading-order checks.

### Task 2: Rewrite the tool guides

- [x] Rewrite `docs/01-aligntune.md`: SFT, preferences, rewards, distillation and adapters; one clearly labeled SFT template; keep PEFT and benchmark caveats.
- [x] Rewrite `docs/02-curatorkit.md`: generation versus formatting, grounding versus truth, grouped splits, provenance and export coverage.
- [x] Rewrite `docs/03-auditkit.md`: task-specific scores, safety versus utility, paired comparisons, cache identities and defective significance results.
- [x] Rewrite `docs/04-safetune.md`: hardening, recovery and live steering; explain directions and score the actual wrapper.
- [x] Rewrite `docs/05-circuitkit.md`: components, attribution versus intervention, faithfulness, Cohere parity and discovery-memory limitations.

### Task 3: Connect ideas and research

- [x] Update `docs/07-project-idea.md`: retain both Aya models and six candidates; recommend two without selecting a final project; replace the stale reading instructions.
- [x] Rewrite `docs/11-community-prior-art.md`: closest multilingual work and useful curation controls, with bounded source-review claims.
- [x] Rewrite `docs/12-neuroai-angles.md`: probes, prediction timing, geometry, matched causal controls and the limits of anomaly features.
- [x] Refocus `docs/14-lexsi-papers.md`: selected relevant papers, questions to bring to each, unchanged checked citation block; remove unrelated tabular and retired project framing.

### Task 4: Verify

- [x] Re-run the checker; expect all edited documents to pass and `13`, `15` and the citation block to match the backup.
- [x] Review word counts and unsupported empirical claims. Report documentation checks only, not unperformed Aya/GPU execution.

**Verification (Nov 2, 2026):** initial run failed 57 checks against the old guides. Final run passed all nine guides, 37 local links, balanced fences and the one Python template's syntax. `13` and `15` match the backup byte-for-byte; the eight-reference BibTeX block is unchanged. A case-sensitive concept check was corrected after confirming that `Random` was already present in `12`. No toolkit imports, model loading or GPU execution occurred.

### Task 5: Align the entry points (Nov 8, 2026)

- [x] Rewrite `README.md`: current reading order, two-model scope, six-day deadline, resources, verification status and archived material.
- [x] Rewrite `DECISION-BRIEF.md`: candidate-1 default with an explicit switch rule, frozen-test definition, day-by-day plan, fallbacks and non-goals.
- [x] Update `docs/00-stack-overview.md`: replace the AgentGuard/Track 3 chains with the current safety/alignment chain, update model roles and remove stale agent-demo guidance.
- [x] Add status banners to `docs/06`, `docs/08` and `docs/09`; they are archived references, not current plans.
- [x] Extend the checker to cover the three entry points and the archive markers; final run passed 63 local links and 12 documents, with `13`/`15` and the BibTeX block still byte-identical.
