# TCN Content SOP Validator — v4 (QA-fixed)

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
QA_REPORT.md                 Full QA report with findings, fixes, and improvement proposals
modules/
  utils.py                   Constants, org-name check, sheet URL builder
  data_loader.py             Google Sheet loading (4 tabs) + anchor/GSC/cannibal helpers
  content_extractor.py       Live-page fetching + draft (paste/.docx) parsing
  embeddings.py              all-MiniLM-L6-v2 loading + paragraph similarity matching
  link_engine.py             Forward + bidirectional suggestion orchestration
  sop_checks.py              All SOP / Koray / link-health checks
```

## Deploy to Streamlit Community Cloud

1. Push this folder to your GitHub repo.
2. In Streamlit Community Cloud: New app → point at the repo → main file `app.py`.
3. `.python-version` (set to `3.12`) handles the Python version.
4. If you want the password gate, add `APP_PASSWORD` under app Settings → Secrets.
5. First run: click **"Build / refresh site content index"** in the sidebar.

## QA-fixed bugs (see QA_REPORT.md for full details)

1. **Paragraph-level matching was broken** — `get_text(" ")` collapsed all pages
   into single blobs. Fixed with leaf-level block element extraction. Index went
   from 10 paragraphs (10 pages) to 365.
2. **Draft FAQ detection broken** — headings excluded from `full_text`. Fixed.
3. **Draft mode false-positive noise** — technical meta/link/image checks ran on
   drafts that can't have them. Fixed with `is_draft` mode.
4. **Suggestion dedup** — same target URL appeared multiple times per paragraph.
   Fixed with (source_paragraph, target_url) dedup.
5. **GSC validation too loose** — single generic words like "cancer" caused false
   matches. Fixed to require 2+ distinct word overlaps.
