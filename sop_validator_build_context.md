# SOP Validator v4 — Build Context

## Who I am
Rejish Shrestha, SEO Specialist at Binaytara (binaytara.org), a 501(c)(3) oncology nonprofit. Nepal-based, remote. I manage SEO for The Cancer News (TCN), IJCCD journal, OncoBlast CME, and 60+ annual conferences. Organization name is always "Binaytara" — never "Binaytara Foundation."

## What exists already
I have a TCN Content SOP Validator Streamlit app (v3) deployed on Streamlit Community Cloud. It checks articles against our Content Publishing SOP — slug, meta title, meta description, headings, internal links, external links, images, content quality, author/trust, technical meta, plus Koray framework checks (hedging language, definitions, title vs H1, thin content, readability) and link health (broken links, response time, redirects).

The SOP checks work fine. **The internal link suggestion engine needs a complete rebuild** — the current version uses TF-IDF on page titles/H1s which produces irrelevant suggestions that don't match the article content contextually.

## The rebuild plan (approved)

### Data sources — all in one Google Sheet
**Spreadsheet ID:** `1T3Hf0gY96o4tPKJH1lvLw94eHxC-kco0CXIJgsDZTkk`
Access via CSV export: `https://docs.google.com/spreadsheets/d/{ID}/gviz/tq?tqx=out:csv&sheet={TAB_NAME}`
Sheet is shared as "anyone with link can view."

**Tabs:**
1. `Internal Linking Anchor Texts` — Columns: Page URL, Page Name, Primary Anchor Text, Alternate Anchor Texts, Top KW Volume, SEO Notes / Warnings. Contains cannibalization warnings (e.g., two ivermectin articles competing), "DO NOT use" anchor rules, and cross-linking instructions.
2. `All Page` — Screaming Frog crawl export (2610 rows). Key columns: Address (URL), Status Code, Indexability ("Indexable"/"Non-Indexable"), Indexability Status, Title 1, H1-1, Meta Description 1. Non-indexable pages must be excluded from suggestions entirely.
3. `GSC Data` — Columns: Page, Query, Clicks, Impressions, CTR, Avg Position. This is the "topic DNA" — what each page actually ranks for in Google.
4. `Semrush Data` — Columns: Keyword, Position, Search Volume, URL, Traffic, Traffic (%), Keyword Intents, etc. Provides keyword volumes and intent data.

### New internal link engine — how it should work

**Step 1: Build site content index (one-time, cached 24h)**
- Fetch `<article>` body text from all indexable TCN and IJCCD pages using 10 parallel threads
- TCN developers use `<article>` tag for article body content
- Compute sentence embeddings using `all-MiniLM-L6-v2` (22MB model, free, runs on Streamlit Cloud free tier)
- Load GSC queries per page as topic validation data
- Load anchor guide as source of truth for anchor text

**Step 2: Match at paragraph level, not page level**
- For each paragraph in the article being checked, find semantically similar paragraphs across all other site pages
- This tells us WHICH paragraph in WHICH page is relevant, not just "this page title kinda matches"
- Use cosine similarity on sentence embeddings

**Step 3: Layer data for validation**
- Anchor guide pages get priority — use their designated primary anchor text
- GSC data validates relevance: does the suggested target page actually rank for queries related to this paragraph's topic?
- Semrush data adds keyword volume context
- Cannibalization warnings from the guide are shown when both competing pages match

**Step 4: Bidirectional suggestions**
- Not just "which pages should THIS article link to" but ALSO "which existing pages should link TO this article"
- For the reverse direction: find existing pages whose paragraphs are semantically similar to this article's topics
- Show the existing paragraph on the source page where the link could be inserted
- Suggest anchor text based on the anchor guide or derived from H1

**Step 5: What each suggestion shows**
- Recommended anchor text (from guide if available, or derived)
- Target URL and page title
- Which sentence in the article to place the link near
- For bidirectional: the existing paragraph on the source page where the link fits
- Confidence/similarity score
- Cannibalization warning if applicable
- GSC validation (does target page rank for related terms?)

### Two modes (keep from v3)
- **Before Publishing**: Writer pastes draft text or uploads .docx. Tool detects structure intelligently (markdown headings, bold-as-headings, Word styles). Optional fields at top: `Slug:`, `Meta Title:`, `Meta Description:`.
- **After Publishing**: Paste live URLs. Tool fetches page, runs all SOP checks + link suggestions.

### SOP checks to keep (all working in v3)
- **Slug**: lowercase, hyphens, no special chars, no IDs, short, minimal stop words
- **Meta title**: 50-60 chars, title case
- **Meta description**: 140-155 chars, no double quotes
- **Headings**: exactly one H1 (50-70 chars), hierarchy (no skipping H2→H4), question-format H2s
- **Internal links**: no first-paragraph links, no duplicate targets, descriptive anchor text (catch "click here"), absolute URLs, no UTM parameters, prefer main domain over subdomains
- **External links**: 0-3 per article, target="_blank"
- **Images**: alt text present, ≤125 chars, no "Image of" prefix, period at end
- **Content quality**: paragraph length (2-4 sentences), FAQ section detection, statistics with sources
- **Author/trust**: byline detection, publication date
- **Technical meta**: OG image (catch localhost bug), canonical URL, robots noindex check, viewport
- **Koray framework**: hedging language detector, definition patterns, title vs H1 relationship, thin content (<300 words), Flesch readability
- **Link health**: broken internal links (HEAD requests), response time, redirect chain detection

### Deployment requirements
- **Platform**: Streamlit Community Cloud (free tier)
- **Python**: 3.12 (must use `.python-version` file — Cloud defaults to 3.14 which breaks greenlet)
- **requirements.txt**: streamlit, requests, beautifulsoup4, lxml, python-docx, scikit-learn, numpy, sentence-transformers
- **No packages.txt needed** — no system-level dependencies (no Playwright/Chromium for this tool)
- **Password gate**: APP_PASSWORD env secret (optional, skipped when not set)
- **Caching**: `@st.cache_data` for Google Sheet data (30 min TTL), `@st.cache_resource` for the embedding model, longer TTL for fetched page content (24h)

### Key rules
- Organization name is always "Binaytara" — never "Binaytara Foundation." Flag if found.
- Non-indexable pages excluded from link suggestions entirely
- When cannibalization warning exists in anchor guide, show both competing pages with the warning
- Anchor guide pages are highest priority suggestions
- TF-IDF fallback for pages not in the anchor guide (but use embeddings, not old TF-IDF)
- Whole team uses this tool: writers check drafts before publishing, developers fix flagged issues, Rejish reviews
- Google Sheet CSV export URL format: `https://docs.google.com/spreadsheets/d/1T3Hf0gY96o4tPKJH1lvLw94eHxC-kco0CXIJgsDZTkk/gviz/tq?tqx=out:csv&sheet={URL_ENCODED_TAB_NAME}`
