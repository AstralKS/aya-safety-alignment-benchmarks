# Dataset Reading Report Implementation Plan

> **For agentic workers:** Execute inline using `executing-plans`; no delegation is needed for this single-file artifact. The workspace is not a Git repository, so worktree creation and commits are unavailable.

**Goal:** Deliver a readable, complete, source-linked offline HTML report of the collected benchmark research and LongFact++ methodology.

**Architecture:** One semantic HTML document contains all text, styling and optional navigation/print enhancement. Existing Markdown remains the source record and is not changed. No build process or external assets are required.

**Tech Stack:** HTML5, inline CSS, small inline JavaScript; Python structural verification and browser inspection when available.

---

### Task 1: Establish the content and verification contract

- [ ] Read `part 1/research/dataset.md` and preserve all benchmark names, release qualifications, dataset counts, independent-use evidence, label-quality cautions, protocols and citations.
- [ ] Include the conversation's additional methodology: original LongFact comparison; prompt generation; preserved entity boundaries; token and span-max loss; optional regularized LoRA; LongFact++ AUROC results; SimpleQA abstention trade-off.
- [ ] Before creation, run `Test-Path -LiteralPath 'D:\ai-sp\part 1\research\dataset.html'`; expected `False`. Do not overwrite an unfamiliar existing file.

### Task 2: Build the single-file report

**Create:** `part 1/research/dataset.html`.

- [ ] Use `<!doctype html>`, `lang="en"`, UTF-8, viewport metadata, a skip link, header, contents navigation, main article, sections and footer.
- [ ] Create sections with IDs `overview`, `labels`, `comparison`, `longfact`, `abstention`, `simpleqa`, `alternatives`, `dates`, `protocol`, `causal`, and `sources`.
- [ ] Use narrative prose and contextual interpretation; supplement rather than replace prose with tables, an illustrative annotated sentence, a methodology sequence and an explicitly hypothetical risk/coverage example.
- [ ] Inline CSS must use readable line lengths, a desktop sidebar, a mobile navigation disclosure, scoped table scrolling, visible focus, reduced-motion support and print styles. No fixed-width content may force page-level horizontal scrolling.
- [ ] Keep all essential content visible without scripting. Add a print button only when JavaScript works, and a guarded `IntersectionObserver` to highlight the current section. Use no network requests or storage.

### Task 3: Verify and deliver

- [ ] Parse the report with Python's `html.parser.HTMLParser`; require unique IDs, working internal anchors, semantic landmarks and all 11 section IDs.
- [ ] Extract every Markdown HTTP(S) URL with `re.findall(r'\]\((https?://[^)]+)\)', text)` and assert each appears in the HTML. Check relative local links with `urllib.parse.unquote` and `pathlib.Path`.
- [ ] Confirm no external scripts, stylesheets, fonts or images are necessary. Check scripts with `node --check` on a temporary extracted script.
- [ ] If Playwright and a browser are available, inspect desktop and mobile at 1440, 1024, 768, 390 and 320px, including page/table overflow, body typography, contents links, print styles and disabled JavaScript. Fix issues and rerun.
- [ ] Confirm the original Markdown hash is unchanged; read back relevant content and final verification output. Final reply links `part 1/research/dataset.html` and mentions the report is offline-readable. Do not claim model experiments or deployment.

## Self-review

The plan covers the approved design, requires no independent subsystem or unrelated modification, and keeps all research qualifications visible. Inline execution is consistent with the user's approval to proceed. Browser checks are conditional on actual tool availability, not assumed to have run.
