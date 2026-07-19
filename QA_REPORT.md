# SOP Validator v4 — QA Report

## Summary

Ran the tool against real live TCN articles and synthetic drafts. Found and
fixed **4 critical bugs** that would have made the tool produce bad or
misleading results in production. The tool is now usable — link suggestions
are semantically meaningful and the SOP checks catch real issues.

---

## Critical Bugs Found and Fixed

### 1. Paragraph-level matching was completely broken (CRITICAL)

**What happened:** `BeautifulSoup.get_text(" ")` strips all newlines, so
every page's article body became ONE giant paragraph. The entire v4
paragraph-level matching engine collapsed to page-level matching — exactly the
same problem as the old TF-IDF approach we were replacing.

**Proof:** Before fix, the stomach cancer article (5,900 chars) produced
1 paragraph. After fix: 23 paragraphs. Across 10 test pages, the index
went from 10 paragraphs to 365.

**Root cause:** `get_text(" ")` joins all text with spaces. The new
`_extract_block_text()` function iterates through leaf-level block elements
(`<p>`, `<li>`, `<h3>`, etc.) individually, skipping container elements
(`<section>`, `<div>`) that would re-absorb their children's text.

**Status:** Fixed and verified.

### 2. Paragraph dedup was too aggressive

Even after the initial block-text fix, nested HTML like `<section><p>...</p></section>`
caused the dedup logic to keep only the `<section>` (which contained all
`<p>` text as substrings), collapsing everything back into one blob.

**Fix:** Replaced substring-based dedup with a "leaf-level only" approach —
only extract from block elements that don't contain nested block children.

**Status:** Fixed and verified.

### 3. Draft mode FAQ detection broken

The `## FAQ` heading was parsed as a heading (correct) but excluded from
`full_text` (bug). The FAQ detection regex searched `full_text` and couldn't
find "FAQ" because heading text wasn't there.

**Fix:** `full_text` now includes heading text for both pasted and .docx
drafts.

**Status:** Fixed and verified.

### 4. Draft mode flooded with false-positive warnings

Technical meta checks (OG image, canonical, viewport), link checks
(internal/external), image checks, and author/trust checks were all running
on draft content — which by definition doesn't have meta tags, rendered
links, or published metadata. Every draft got 5+ yellow/red warnings that
were noise, not actionable.

**Fix:** Added `is_draft` mode flag. In "Before Publishing" mode, only
checks that apply to draft content run: org name, slug, meta title, meta
description, headings, content quality, Koray framework (hedging, definitions,
title vs H1, thin content, readability).

**Status:** Fixed and verified.

---

## Other Issues Fixed (from initial build)

- **Cannibalization false positives**: 10 rows matched instead of 4 because
  notes saying "✅ No cannibalization" still contained "cannibalization."
  Fixed to require ⚠️ warning marker.
- **Anchor ban parsing**: "DO NOT use X or Y as anchor" clauses with
  multiple quoted terms now capture all banned phrases, not just the first.
- **Slug year false positive**: `key-ash-2025-updates` was flagged as
  containing an "ID number." Fixed to exempt plausible years (1900-2099).
- **Section divider rows**: 3 non-URL rows (`══ HIGH-TRAFFIC... ══`) in the
  anchor guide were being treated as real pages. Filtered out.

---

## Suggestion Quality Assessment

**Test: Stomach cancer article against 10-page index**

| Source paragraph | Target suggestion | Anchor text | Similarity | SEO verdict |
|---|---|---|---|---|
| "Anemia, especially iron deficiency anemia (IDA)..." | Low Hemoglobin article | "what type of cancer causes low hemoglobin" | 0.715 | ✅ Correct — anemia discussion should link to the dedicated hemoglobin article |
| Expert quote about symptoms/black stool | Low Hemoglobin article | same | 0.583 | ✅ Correct — symptom overlap is genuine |
| "On Signs/Symptoms of Stomach Cancer" | Metastatic Cancer article | "metastatic cancer" | 0.564 | ⚠️ Borderline — the semantic match is weak, but the GSC validates it |

**Bidirectional:** The Low Hemoglobin article's paragraphs about "anemia in
cancer patients" correctly suggested linking back to the stomach cancer
article with anchor text "stomach cancer symptoms." This is a real actionable
suggestion a writer could implement today.

**Verdict:** Suggestion quality is **good enough to ship** for the team's
workflow. With the full ~1,700-page index (not the 10-page test index), the
suggestions will be much richer.

---

## Org Name Flag

Row 23 in your Semrush data: **"binaytara foundation" ranks #1 with 170
monthly search volume.** People are actively searching the old name.
Consider either:
- A 301 redirect strategy for `/binaytara-foundation` URLs to their
  corrected equivalents
- Ensuring the homepage/about page naturally captures this traffic without
  reinforcing the wrong name

The anchor guide already flags the URL
`binaytara.org/news/health/binaytara-foundation-s-5-year-strategic-plan-...`
with a redirect recommendation. The app surfaces this in the sidebar on load.

---

## Screaming Frog .seospider File

**Can I read it?** A `.seospider` file is a SQLite database — technically
readable. But it can't be uploaded to this chat (too large for the upload
limit), and Streamlit Community Cloud can't access your local `D:\` drive.

**Practical options for integrating Screaming Frog data:**

1. **Current approach (recommended):** Export the crawl data to CSV from
   Screaming Frog, paste it into your "All Page" sheet tab. This is what
   you're already doing with the 2,610-row export. Keep doing this — update
   it after each crawl.

2. **If you want live crawl data in the tool:** You could export the SF
   crawl to a CSV file, upload it to Google Drive, and have the tool read it
   from there. But this adds complexity for minimal gain — the crawl data
   changes infrequently (you recrawl monthly), so a manual sheet update is
   fine.

3. **If you want specific SF columns the sheet doesn't currently have:**
   Export them from SF and add new columns to the "All Page" tab. The tool
   can read any column you add.

---

## Sheet Data Improvements You Can Make Now

These are things you can update manually in the existing Google Sheet to get
better suggestions immediately — no code changes needed:

### Anchor Guide tab ("Internal Linking Anchor Texts")

1. **Add more pages.** Currently 35 rows. Your top 20+ traffic articles
   should all be in here. Any page getting 50+ monthly organic clicks
   deserves an anchor-guide entry.

2. **Review anchor text length.** The low-hemoglobin article's primary
   anchor is "what type of cancer causes low hemoglobin" (47 chars) — that's
   extremely long for inline anchor text. Consider shorter alternatives like
   "cancers that cause low hemoglobin" or "low hemoglobin and cancer."

3. **Add cross-linking instructions.** For articles that naturally relate
   (e.g. stomach cancer ↔ low hemoglobin ↔ metastatic cancer), add notes
   like "Should link to/from: [URL]" in the SEO Notes column. The tool
   surfaces these notes in the suggestion output.

### All Page tab

4. **Re-export after each crawl.** The 2,610-row export is your indexability
   source of truth. After each Screaming Frog crawl, re-export and replace
   the tab data. Stale data = the tool suggests linking to pages that no
   longer exist or are now noindexed.

### GSC Data tab

5. **Expand the query set.** The tool uses GSC queries as "topic DNA"
   validation. More queries = better validation. Currently if a page has <5
   queries in the sheet, the GSC validation signal is weak. Consider exporting
   a broader date range or lower click threshold from GSC.

### Semrush Data tab

6. **The Semrush data is clean.** 99 rows, all typed correctly. No action
   needed, but keep it updated monthly alongside the anchor guide.

---

## Proposed Improvements (Need Your Approval)

These are on-page SEO checks and features I can add to the tool. None are
built yet — tell me which ones you want and I'll build them.

### A. Word count by section + content depth scoring

Currently the tool only checks total word count (>300 words). A more useful
check: compare word count per H2 section against competitors for the same
target keyword. Thin individual sections are a bigger problem than thin total
word count.

**What it would do:** Flag H2 sections under 100 words. Show word count per
section in the output so writers know where to expand.

### B. Keyword in H1 / H2 / first paragraph check

The tool checks heading structure (hierarchy, length, questions) but doesn't
check whether the target keyword actually appears in the H1, first H2, or
opening paragraph — a fundamental on-page signal.

**What it would do:** Accept a target keyword field in the UI (optional).
When provided, check H1 contains it, first paragraph contains it, at least
one H2 contains a variation.

### C. Content gap detection vs. top-ranking competitors

For "After Publishing" mode: given the article's target keyword, fetch the
top 3-5 SERP competitors and compare heading structures. Flag H2 topics the
competitors cover that this article doesn't — these are content gaps that
could explain ranking plateaus.

**What it would do:** Show "Competitor X covers [topic] under H2 '[heading]'
but your article doesn't." Requires web fetching competitors at check time.

### D. Schema markup validation

The tool currently checks basic meta tags but doesn't validate JSON-LD
schema. For TCN articles, it should check for Article/NewsArticle schema
with required fields (headline, datePublished, author, publisher, image).

**What it would do:** Parse JSON-LD from live pages, flag missing required
fields, flag mismatches between schema fields and visible page content.

### E. Internal link density scoring

Instead of just checking link presence/absence, calculate internal links per
1,000 words and compare against your own site average. Flag articles that are
significantly below average.

**What it would do:** Show "This article has 0.8 internal links per 1,000
words vs. site average of 2.3 — consider adding more."

### F. Image optimization checks (WebP format, aspect ratios)

Your SOP specifies WebP format and three aspect ratios for Google Discover
eligibility. The tool currently only checks alt text. It should also check
image format (is it WebP?) and dimensions (does it meet Discover's
1200px-wide minimum?).

**What it would do:** Flag non-WebP images, flag images under 1200px wide,
check aspect ratio against the three SOP-specified ratios.

### G. Semrush keyword volume display in suggestions

The tool has Semrush data loaded but doesn't currently show search volume
alongside link suggestions. Showing "Target page ranks for 'stomach cancer
symptoms' (90,500 monthly)" next to a suggestion gives writers instant
context on the SEO value of adding that link.

**What it would do:** Add search volume badge to each link suggestion.
No new code needed for data — just wiring the Semrush tab into the
suggestion rendering.

### H. Export check results to CSV/clipboard

Writers and developers need to act on the findings outside the tool —
copying results into a Slack message or task tracker. A one-click "Export
results" button that produces a clean CSV or copies to clipboard would save
time.

---

**Which improvements do you want me to build?** I can do any combination.
G and H are the quickest wins (hours, not days). C is the most ambitious.
