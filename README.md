# TCN Content SOP Validator — v4

Rebuild of the internal link suggestion engine: paragraph-level semantic
matching (sentence embeddings) instead of TF-IDF on titles/H1s, layered with
your anchor guide, GSC "topic DNA," and cannibalization warnings — plus
bidirectional suggestions (which existing pages should link back to a new
article). All v3 SOP / Koray / link-health checks are kept.

## Files

```
app.py                       Main Streamlit app (two modes: Before/After Publishing)
.python-version              Pins Python 3.12 (Cloud defaults to 3.14, breaks greenlet)
requirements.txt
modules/
  utils.py                   Constants, org-name check, sheet URL builder
  data_loader.py             Google Sheet loading (4 tabs) + anchor/GSC/cannibal helpers
  content_extractor.py       Live-page fetching + draft (paste/.docx) parsing
  embeddings.py              all-MiniLM-L6-v2 loading + paragraph similarity matching
  link_engine.py             Forward + bidirectional suggestion orchestration
  sop_checks.py              All SOP / Koray / link-health checks
```

## Deploy to Streamlit Community Cloud

1. Push this folder to your GitHub repo (new repo, or overwrite the v3 one —
   your call; I didn't have write access to push it for you).
2. In Streamlit Community Cloud: New app → point at the repo → main file `app.py`.
3. `.python-version` (already set to `3.12`) takes care of the Python version.
4. If you want the password gate, add `APP_PASSWORD` under app Settings → Secrets:
   ```
   APP_PASSWORD = "your-password-here"
   ```
   Leave it unset to skip the gate entirely.
5. First run: click **"Build / refresh site content index"** in the sidebar.
   This fetches all ~1,700 indexable pages (10 parallel threads) and computes
   embeddings — takes a few minutes the first time, then it's cached 24h.

## What I verified against your live data

- Google Sheet CSV export works against your real spreadsheet
  (`1T3Hf0gY96o4tPKJH1lvLw94eHxC-kco0CXIJgsDZTkk`) — all four tabs load with
  the exact column names your sheet actually uses.
- The **Internal Linking Anchor Texts** tab has 3 non-URL "section divider"
  rows (e.g. `══ HIGH-TRAFFIC CANCER NEWS ARTICLES ══`) — filtered out
  automatically so they don't get treated as real pages.
- Cannibalization detection: your sheet uses ✅ for all-clear and ⚠️ for
  warnings, and several ✅ notes literally contain the word "cannibalization"
  (e.g. "No cannibalization"). Fixed the parser to require the warning marker
  — it now correctly finds the 4 real cannibalization pairs (ivermectin ×2,
  the James Pickens Jr. page, and the 2025 year-in-review vs. top-innovations
  pages) instead of 10 false positives.
- "DO NOT use ... as anchor" bans (e.g. "ivermectin and cancer" / "ivermectin
  for cancer", "James Pickens Jr") are now parsed and the link engine will
  never recommend a banned anchor — it falls back to an alternate anchor from
  the guide, or a derived one.
- **Org name flag**: your anchor guide already has a row flagging
  `binaytara-foundation` in a URL slug:
  `https://binaytara.org/news/health/binaytara-foundation-s-5-year-strategic-plan-...`
  — the sidebar now surfaces this automatically on load, plus every check run
  scans the article text/meta/slug for "Binaytara Foundation" and fails loudly
  if found.
- Confirmed `<article>` tag extraction against a real live TCN page (ASH 2025
  NHL article) — pulled clean body text, correct title/H1, and (accurately)
  found zero internal links in that particular article, which is exactly the
  kind of gap this tool exists to catch.
- Fixed a slug-check false positive: years like "2025" inside a slug
  (`key-ash-2025-updates-...`) were being flagged as "ID-like numbers" —
  now only genuinely ID-like numbers get flagged.
- Full pipeline test (embeddings → anchor guide → GSC validation) confirmed
  paragraphs about "stage 4 cancer" and "stomach cancer symptoms" correctly
  matched their real target pages with the right guide anchor text and GSC
  validation.
- Booted the actual Streamlit app locally (HTTP 200, clean logs) before
  handing this off.

## Known constraints / things to watch after deploying

- First index build fetches ~1,700 pages — if Streamlit Cloud's free tier
  times out on the initial build, consider lowering `MAX_WORKERS` in
  `content_extractor.py` or pre-warming the cache via a scheduled job.
- `sentence-transformers` pulls in `torch`, which is the heaviest dependency
  here — if the free tier's build fails on size/memory, the fallback is to
  swap to `onnxruntime`-based embeddings, but I'd only make that change if you
  hit the wall in practice.
- The Semrush "Traffic (%)"/"CTR" style percentage columns are strings in the
  sheet (e.g. `"0.48%"`) — none of the current checks parse them numerically,
  so no bug there, but flagging in case you build on top of this.
