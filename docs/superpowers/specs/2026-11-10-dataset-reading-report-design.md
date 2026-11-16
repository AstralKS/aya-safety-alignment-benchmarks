# Dataset research HTML report — design

## Approved direction

The user chose an editorial research report rather than a minimal document or comparison-first dashboard. Deliver `part 1/research/dataset.html` as a self-contained offline page. Preserve `part 1/research/dataset.md` and all existing project files.

## Purpose and content

Make the collected dataset research understandable through complete paragraphs, explicit reasoning, concrete examples, and qualified evidence. Cover every benchmark and evaluation detail in `dataset.md`, plus the LongFact++ methodology discussed in the conversation. This is a reading artifact, not dataset selection or experimental implementation.

Organize the page into: recommendation and rationale; what benchmark labels measure; comparison and independent adoption; LongFact++ construction, annotation, probe loss, transfer, results and limitations; AbstentionBench; SimpleQA Verified; other candidates; date exclusions; proposed Aya protocol, metrics and causal handoff; references. Distinguish observations, published results and proposed experiments. Preserve original release-count and date qualifications. Link all sources from the Markdown and methodology discussion. Include links back to the original Markdown and companion research files.

## Presentation and architecture

One semantic HTML file with inline CSS and small optional JavaScript; no framework, external fonts, downloaded assets, network calls or build step. Warm paper, dark ink, restrained teal and rust accents, Georgia headings and readable local sans-serif body fonts. Target 65–85 characters per prose line, 17–18px body text, generous line height, and normal word wrapping. Use a desktop contents sidebar and a mobile contents disclosure. Main text stays in substantial paragraphs; tables scroll within their containers on narrow screens. All important content is visible without JavaScript.

Use native anchors, meaningful headings, accessible tables with captions and header scopes, visible focus styles and a skip link. Optional JavaScript updates the current contents item and enables a print button; print remains available from the browser without it. Printing removes navigation and decorative UI while keeping report text and sources. Respect reduced motion. No essential content hidden in tabs or cards.

## Verification and error handling

Check HTML structure, unique section IDs, every local anchor, reference coverage and local file links. Where browser tooling is available, inspect desktop and mobile widths for overflow, text width, typography, navigation and printing. Report unavailable browser verification honestly. External citations open normally; the page itself requires no network connection. No user settings, storage or asynchronous data loading.

## Scope and review

No new Aya results, deployment, dataset downloads, training, images or changes to existing research. The workspace is not a Git repository, so no commit can be made. Review found no placeholders, contradictory requirements or independent subsystems requiring decomposition.
