"""
TCN Content SOP Validator v4 — single-file build
=================================================
Everything in one file so deployment can never break on folder structure.
Sections: Utilities -> Sheet loading -> Content extraction -> Embeddings ->
Link engine -> SOP checks -> Streamlit UI.
"""

# --- Standard library ---
import io
import os
import re
import concurrent.futures
import urllib.parse

# --- Third-party ---
import numpy as np
import pandas as pd
import requests
import streamlit as st
import textstat
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# =============================================================================
# SECTION 1: UTILITIES & CONSTANTS
# =============================================================================

"""Shared constants and small helpers for the SOP Validator."""

SPREADSHEET_ID = "1T3Hf0gY96o4tPKJH1lvLw94eHxC-kco0CXIJgsDZTkk"

TAB_ANCHOR_GUIDE = "Internal Linking Anchor Texts"
TAB_ALL_PAGE = "All Page"
TAB_GSC = "GSC Data"
TAB_SEMRUSH = "Semrush Data"

ORG_NAME_CORRECT = "Binaytara"
ORG_NAME_WRONG = "Binaytara Foundation"

USER_AGENT = "Mozilla/5.0 (compatible; BinaytaraSOPValidator/4.0; +https://binaytara.org)"


def sheet_csv_url(spreadsheet_id: str, tab_name: str) -> str:
    """Build the gviz CSV export URL for a given tab name."""
    encoded_tab = urllib.parse.quote(tab_name)
    return (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        f"/gviz/tq?tqx=out:csv&sheet={encoded_tab}"
    )


def find_org_name_violations(text: str) -> list:
    """Flag every occurrence of 'Binaytara Foundation' in a block of text.

    Returns a list of short context snippets so the user can locate each hit.
    """
    if not text:
        return []
    hits = []
    for m in re.finditer(re.escape(ORG_NAME_WRONG), text, flags=re.IGNORECASE):
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        snippet = text[start:end].replace("\n", " ").strip()
        hits.append(f"...{snippet}...")
    return hits


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_into_paragraphs(article_text: str) -> list:
    """Split article body text into non-trivial paragraphs (>= 25 chars)."""
    raw_paras = re.split(r"\n\s*\n", article_text or "")
    paras = [clean_whitespace(p) for p in raw_paras]
    return [p for p in paras if len(p) >= 25]


def domain_of(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""


# =============================================================================
# SECTION 2: GOOGLE SHEET DATA LOADING
# =============================================================================

"""Loads the four Google Sheet tabs that power the SOP Validator.

All tabs live in one spreadsheet, shared as "anyone with link can view",
and are pulled via the gviz CSV export endpoint (no API key / OAuth needed).
"""


SHEET_TTL_SECONDS = 30 * 60  # 30 min, per spec


def _fetch_csv(tab_name: str) -> pd.DataFrame:
    url = sheet_csv_url(SPREADSHEET_ID, tab_name)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # Normalize column names: strip whitespace so lookups are predictable
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data(ttl=SHEET_TTL_SECONDS, show_spinner=False)
def load_anchor_guide() -> pd.DataFrame:
    """Internal Linking Anchor Texts tab.

    Expected columns: Page URL, Page Name, Primary Anchor Text,
    Alternate Anchor Texts, Top KW Volume, SEO Notes / Warnings
    """
    df = _fetch_csv(TAB_ANCHOR_GUIDE)
    for col in ["Page URL", "Page Name", "Primary Anchor Text",
                "Alternate Anchor Texts", "Top KW Volume", "SEO Notes / Warnings"]:
        if col not in df.columns:
            df[col] = ""
    df["Page URL"] = df["Page URL"].astype(str).str.strip()
    # The sheet uses non-URL rows as visual section dividers (e.g. "══ ... ══")
    # — drop those so they don't get treated as real anchor-guide entries.
    df = df[df["Page URL"].str.startswith("http")].reset_index(drop=True)
    return df


@st.cache_data(ttl=SHEET_TTL_SECONDS, show_spinner=False)
def load_all_pages() -> pd.DataFrame:
    """All Page tab — Screaming Frog crawl export.

    Expected columns include: Address, Status Code, Indexability,
    Indexability Status, Title 1, H1-1, Meta Description 1
    """
    df = _fetch_csv(TAB_ALL_PAGE)
    if "Address" in df.columns:
        df["Address"] = df["Address"].astype(str).str.strip()
    if "Indexability" not in df.columns:
        df["Indexability"] = "Indexable"
    return df


@st.cache_data(ttl=SHEET_TTL_SECONDS, show_spinner=False)
def load_gsc_data() -> pd.DataFrame:
    """GSC Data tab — Page, Query, Clicks, Impressions, CTR, Avg Position."""
    df = _fetch_csv(TAB_GSC)
    if "Page" in df.columns:
        df["Page"] = df["Page"].astype(str).str.strip()
    return df


@st.cache_data(ttl=SHEET_TTL_SECONDS, show_spinner=False)
def load_semrush_data() -> pd.DataFrame:
    """Semrush Data tab — Keyword, Position, Search Volume, URL, Traffic, etc."""
    df = _fetch_csv(TAB_SEMRUSH)
    if "URL" in df.columns:
        df["URL"] = df["URL"].astype(str).str.strip()
    return df


def get_indexable_urls(all_pages_df: pd.DataFrame) -> list:
    """Return only indexable, 200-status URLs, ready for the content index."""
    df = all_pages_df.copy()
    mask = df["Indexability"].astype(str).str.strip().str.lower() == "indexable"
    if "Status Code" in df.columns:
        mask &= pd.to_numeric(df["Status Code"], errors="coerce").fillna(0).astype(int) == 200
    return df.loc[mask, "Address"].dropna().unique().tolist()


HUB_PAGE_PATTERNS = (
    "/cancernews/all-articles", "/cancernews/research", "/cancernews/cancer-education",
    "/cancernews/authors", "/cancernews/contributors",
)


def is_hub_or_nav_page(url: str) -> bool:
    """Hub/listing pages contain article teasers (duplicated titles/intros),
    which caused false 1.0-similarity matches. They're linked from nav anyway,
    so they're never useful in-content link suggestions."""
    u = url.rstrip("/")
    if u in ("https://binaytara.org", "http://binaytara.org"):
        return True
    if u.endswith("/cancernews"):
        return True
    return any(p in u for p in HUB_PAGE_PATTERNS)


def is_content_article_url(url: str) -> bool:
    """Only real article pages produce meaningful paragraph matches."""
    return "/cancernews/article/" in url or "/ijccd" in url


def gsc_queries_for_page(gsc_df: pd.DataFrame, page_url: str, top_n: int = 15) -> list:
    """Top queries (by clicks) that a given page ranks for — the 'topic DNA'."""
    if gsc_df.empty or "Page" not in gsc_df.columns:
        return []
    sub = gsc_df[gsc_df["Page"] == page_url]
    if sub.empty or "Query" not in sub.columns:
        return []
    sort_col = "Clicks" if "Clicks" in sub.columns else sub.columns[0]
    sub = sub.sort_values(sort_col, ascending=False)
    return sub["Query"].dropna().astype(str).head(top_n).tolist()


def anchor_entry_for_url(anchor_df: pd.DataFrame, page_url: str):
    """Return the anchor-guide row for a URL, or None if not in the guide."""
    if anchor_df.empty:
        return None
    match = anchor_df[anchor_df["Page URL"] == page_url]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def cannibalization_pairs(anchor_df: pd.DataFrame) -> dict:
    """Parse 'SEO Notes / Warnings' for genuine cannibalization callouts.

    The sheet's convention is a ⚠️ prefix for warnings and ✅ for all-clear —
    notes like "✅ No cannibalization" must NOT match here, so we require an
    explicit warning marker alongside the word, not just the substring.
    Returns {page_url: warning_text}.
    """
    out = {}
    if anchor_df.empty:
        return out
    for _, row in anchor_df.iterrows():
        note = str(row.get("SEO Notes / Warnings", "")).strip()
        url = str(row.get("Page URL", "")).strip()
        if not (url and note and note.lower() != "nan"):
            continue
        has_warning_marker = "⚠️" in note or "CANNIBALIZATION" in note.upper()
        says_no_cannibalization = bool(re.search(r"no\s+cannibalization", note, re.IGNORECASE))
        if "cannibal" in note.lower() and has_warning_marker and not says_no_cannibalization:
            out[url] = note
    return out


def anchor_bans_for_url(anchor_df: pd.DataFrame, page_url: str) -> list:
    """Parse 'DO NOT use "X" [or "Y"] as anchor' style rules out of the
    notes for a page, so the link engine never recommends a banned anchor.
    Handles multiple quoted terms chained with "or" in one clause.
    """
    entry = anchor_entry_for_url(anchor_df, page_url)
    if not entry:
        return []
    note = str(entry.get("SEO Notes / Warnings", ""))
    bans = []
    for clause_match in re.finditer(r"DO NOT use\s+(.*?)(?:as anchor|$)", note, re.IGNORECASE):
        clause = clause_match.group(1)
        bans.extend(re.findall(r'"([^"]+)"', clause))
    return bans


def org_name_flags_in_sheet(anchor_df: pd.DataFrame) -> list:
    """Surface any anchor-guide rows that themselves flag 'Binaytara
    Foundation' / 'binaytara-foundation' issues (e.g. a legacy URL slug),
    so the app can call these out to the user proactively.
    """
    hits = []
    if anchor_df.empty:
        return hits
    for _, row in anchor_df.iterrows():
        note = str(row.get("SEO Notes / Warnings", ""))
        url = str(row.get("Page URL", ""))
        combined = f"{url} {note}"
        if "binaytara foundation" in combined.lower() or "binaytara-foundation" in combined.lower():
            hits.append({"url": url, "note": note})
    return hits


# =============================================================================
# SECTION 3: CONTENT EXTRACTION (live pages + drafts)
# =============================================================================

"""Content extraction:
1. Fetch <article> body text from live TCN/IJCCD pages (site content index).
2. Parse pasted text / uploaded .docx drafts for the "Before Publishing" mode,
   detecting structure from markdown headings, bold-as-headings, or Word styles.
"""


BLOCK_TAGS = {"p", "li", "blockquote", "figcaption", "td", "th",
              "h1", "h2", "h3", "h4", "h5", "h6", "div", "section"}


def _extract_block_text(element) -> str:
    """Walk an element and return paragraph-separated text.

    Extracts text from leaf-level block elements (like <p>, <li>, <h2>) — i.e.
    block tags that don't contain other block tags. This avoids the problem
    where a <section> containing 16 <p> tags would collapse into one giant
    blob, defeating paragraph-level matching.
    """
    if element is None:
        return ""
    parts = []
    for child in element.descendants:
        if child.name not in BLOCK_TAGS:
            continue
        # Skip this block if it contains nested block-level children —
        # we want the innermost blocks (the actual paragraphs).
        has_nested_block = any(
            desc.name in BLOCK_TAGS for desc in child.descendants if hasattr(desc, "name")
        )
        if has_nested_block:
            continue
        text = clean_whitespace(child.get_text(" "))
        if text and len(text) >= 20:
            parts.append(text)
    # If no leaf-level blocks found (unusual HTML), fall back to full text
    if not parts:
        text = clean_whitespace(element.get_text(" "))
        if text:
            parts.append(text)
    return "\n\n".join(parts)


FETCH_TIMEOUT = 15
MAX_WORKERS = 10
CONTENT_TTL_SECONDS = 24 * 60 * 60  # 24h, per spec


def _extract_article_html(html: str):
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article")
    if article is None:
        # fall back to <main> then the whole body, better a rough index
        # entry than nothing
        article = soup.find("main") or soup.body
    return article, soup


def extract_page_details(html: str, page_url: str) -> dict:
    """Pull everything the SOP checks need out of a live page's HTML:
    headings, links (internal/external), images, meta tags, byline hints.
    """
    import urllib.parse

    soup = BeautifulSoup(html, "lxml")
    article, _ = _extract_article_html(html)
    scope = article if article else soup

    # Headings in document order, within the article scope
    headings = []
    for tag in scope.find_all(re.compile(r"^h[1-6]$")):
        level = int(tag.name[1])
        text = clean_whitespace(tag.get_text())
        if text:
            headings.append((level, text))

    page_domain = domain_of(page_url)
    links = []
    for a in scope.find_all("a", href=True):
        href = a["href"]
        absolute = urllib.parse.urljoin(page_url, href)
        is_internal = domain_of(absolute) == page_domain or domain_of(absolute) == ""
        links.append({
            "href": absolute,
            "anchor_text": clean_whitespace(a.get_text()),
            "is_internal": is_internal,
            "target": a.get("target"),
        })

    images = []
    for img in scope.find_all("img"):
        images.append({"src": img.get("src", ""), "alt": img.get("alt", "")})

    def meta_content(**attrs):
        tag = soup.find("meta", attrs=attrs)
        return tag.get("content") if tag else None

    canonical_tag = soup.find("link", rel="canonical")
    paragraphs_text = _extract_block_text(scope)

    # Publication date: JSON-LD datePublished is the most reliable source,
    # falling back to a <time> tag, then a visible "Month DD, YYYY" string.
    pub_date = None
    m_date = re.search(r'"datePublished"\s*:\s*"([^"]+)"', html)
    if m_date:
        pub_date = m_date.group(1)
    elif soup.find("time"):
        t = soup.find("time")
        pub_date = t.get("datetime") or clean_whitespace(t.get_text())
    else:
        m_vis = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            paragraphs_text)
        if m_vis:
            pub_date = m_vis.group(0)

    return {
        "headings": headings,
        "links": links,
        "images": images,
        "meta_title": clean_whitespace(soup.find("title").get_text()) if soup.find("title") else "",
        "meta_description": meta_content(name="description") or "",
        "og_image": meta_content(property="og:image"),
        "canonical": canonical_tag.get("href") if canonical_tag else None,
        "robots": meta_content(name="robots"),
        "viewport": meta_content(name="viewport"),
        "full_text": paragraphs_text,
        "pub_date": pub_date,
        "h1": (lambda hs: hs[0][1] if hs else "")([h for h in headings if h[0] == 1]),
    }


def fetch_page(url: str) -> dict:
    """Fetch a single URL and pull out article text + basic meta.

    Returns a dict; on failure 'ok' is False and 'error' explains why.
    """
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=FETCH_TIMEOUT)
        status = resp.status_code
        if status != 200:
            return {"url": url, "ok": False, "status": status, "error": f"HTTP {status}"}
        article, soup = _extract_article_html(resp.text)
        text = _extract_block_text(article) if article else ""
        title_tag = soup.find("title")
        h1_tag = soup.find("h1")
        return {
            "url": url,
            "ok": True,
            "status": status,
            "text": text,
            "title": clean_whitespace(title_tag.get_text()) if title_tag else "",
            "h1": clean_whitespace(h1_tag.get_text()) if h1_tag else "",
            "raw_html_len": len(resp.text),
        }
    except requests.exceptions.RequestException as e:
        return {"url": url, "ok": False, "status": None, "error": str(e)}


@st.cache_data(ttl=CONTENT_TTL_SECONDS, show_spinner=False)
def build_site_content_index(urls: tuple) -> list:
    """Fetch article bodies for many URLs in parallel. Cached 24h.

    `urls` must be a tuple (not a list) so Streamlit can hash it for caching.
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_page, u): u for u in urls}
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    # keep original order stable-ish for reproducibility
    order = {u: i for i, u in enumerate(urls)}
    results.sort(key=lambda r: order.get(r["url"], 0))
    return results


# ---------------------------------------------------------------------------
# Draft parsing (Before Publishing mode)
# ---------------------------------------------------------------------------

FRONT_MATTER_FIELDS = {"slug", "meta title", "meta description"}


def parse_front_matter(raw_text: str) -> tuple:
    """Pull optional `Slug:` / `Meta Title:` / `Meta Description:` lines from
    the top of a pasted draft. Returns (front_matter_dict, remaining_text).
    """
    lines = raw_text.splitlines()
    front = {}
    consumed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            consumed += 1
            continue
        m = re.match(r"^(slug|meta title|meta description)\s*:\s*(.+)$", stripped, re.IGNORECASE)
        if m:
            front[m.group(1).lower()] = m.group(2).strip()
            consumed += 1
        else:
            break
    remaining = "\n".join(lines[consumed:])
    return front, remaining


def parse_pasted_draft(raw_text: str) -> dict:
    """Detect headings in pasted plain/markdown text.

    Heuristics: '#'-style markdown headings, or a short standalone line in
    ALL CAPS / Title Case followed by a blank line (treated as bold-as-heading
    when no markdown is present).
    """
    front_matter, body = parse_front_matter(raw_text)
    lines = body.splitlines()
    headings = []  # (level, text, line_index)
    paragraphs = []
    buffer = []

    def flush_buffer():
        if buffer:
            text = clean_whitespace(" ".join(buffer))
            if text:
                paragraphs.append(text)
            buffer.clear()

    for line in lines:
        stripped = line.strip()
        md_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if md_match:
            flush_buffer()
            level = len(md_match.group(1))
            headings.append((level, clean_whitespace(md_match.group(2))))
            continue
        # bold-as-heading: **Heading** alone on its own line
        bold_match = re.match(r"^\*\*(.+?)\*\*$", stripped)
        if bold_match and len(stripped) < 100:
            flush_buffer()
            headings.append((2, clean_whitespace(bold_match.group(1))))
            continue
        if stripped == "":
            flush_buffer()
            continue
        buffer.append(stripped)
    flush_buffer()

    # Include heading text in full_text so keyword checks (FAQ, definition
    # patterns) work — headings are where "FAQ" and definitions often live.
    all_blocks = [text for _, text in headings] + paragraphs
    full_text = "\n\n".join(all_blocks)
    h1_candidates = [h for lvl, h in headings if lvl == 1]

    return {
        "front_matter": front_matter,
        "headings": headings,
        "paragraphs": paragraphs,
        "full_text": full_text,
        "h1": h1_candidates[0] if h1_candidates else None,
    }


def parse_docx_draft(file_bytes) -> dict:
    """Parse an uploaded .docx using python-docx, reading Word paragraph
    styles (Heading 1, Heading 2, ...) to detect structure.
    """
    from docx import Document
    import io as _io

    doc = Document(_io.BytesIO(file_bytes))
    headings = []
    paragraphs = []
    front_matter = {}

    for para in doc.paragraphs:
        text = clean_whitespace(para.text)
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        m = re.match(r"^(slug|meta title|meta description)\s*:\s*(.+)$", text, re.IGNORECASE)
        if m and not front_matter:
            front_matter[m.group(1).lower()] = m.group(2).strip()
            continue
        heading_match = re.match(r"^heading\s*(\d)$", style)
        if heading_match:
            headings.append((int(heading_match.group(1)), text))
        else:
            paragraphs.append(text)

    all_blocks = [text for _, text in headings] + paragraphs
    full_text = "\n\n".join(all_blocks)
    h1_candidates = [h for lvl, h in headings if lvl == 1]

    return {
        "front_matter": front_matter,
        "headings": headings,
        "paragraphs": paragraphs,
        "full_text": full_text,
        "h1": h1_candidates[0] if h1_candidates else None,
    }


# =============================================================================
# SECTION 4: EMBEDDINGS & SEMANTIC MATCHING
# =============================================================================

"""Embedding model + paragraph-level semantic matching.

Uses all-MiniLM-L6-v2 (22MB, free, runs fine on Streamlit Community Cloud's
free tier) to embed paragraphs and match them by cosine similarity — this
replaces the old TF-IDF-on-title/H1 approach with real content-level matching.
"""


MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed_paragraphs(paragraphs: list) -> np.ndarray:
    """Embed a list of paragraph strings. Returns an (n, d) array."""
    if not paragraphs:
        return np.zeros((0, 384))
    model = load_model()
    return model.encode(paragraphs, show_progress_bar=False, convert_to_numpy=True)


def build_paragraph_index(site_content: list) -> dict:
    """Flatten every page's article text into paragraphs, embed them all,
    and keep a lookup back to (url, title, h1, paragraph_index, text).

    Returns {"embeddings": np.ndarray, "meta": [ {url, title, h1, para_idx, text} ]}
    """
    meta = []
    all_paragraphs = []
    for page in site_content:
        if not page.get("ok") or not page.get("text"):
            continue
        paras = [p for p in split_into_paragraphs(page["text"]) if len(p) >= 120]
        for i, p in enumerate(paras):
            meta.append({
                "url": page["url"],
                "title": page.get("title", ""),
                "h1": page.get("h1", ""),
                "para_idx": i,
                "text": p,
            })
            all_paragraphs.append(p)

    embeddings = embed_paragraphs(all_paragraphs)
    return {"embeddings": embeddings, "meta": meta}


def match_paragraph_against_index(paragraph_embedding: np.ndarray, index: dict,
                                    exclude_url: str = None, top_k: int = 5,
                                    min_similarity: float = 0.55,
                                    max_similarity: float = 0.93) -> list:
    """Find the top-k most similar indexed paragraphs to a single query
    paragraph embedding, optionally excluding a source URL (so an article
    doesn't get matched against itself when it's already published).
    """
    embeddings = index["embeddings"]
    meta = index["meta"]
    if embeddings.shape[0] == 0:
        return []

    sims = cosine_similarity(paragraph_embedding.reshape(1, -1), embeddings)[0]
    ranked_idx = np.argsort(-sims)

    results = []
    for idx in ranked_idx:
        if len(results) >= top_k:
            break
        m = meta[idx]
        if exclude_url and m["url"] == exclude_url:
            continue
        score = float(sims[idx])
        if score < min_similarity:
            break  # sims sorted descending, so we can stop early
        if score >= max_similarity:
            # Near-identical text = the same teaser/snippet on another page,
            # not a genuine linking opportunity. Skip.
            continue
        results.append({**m, "similarity": score})
    return results


# =============================================================================
# SECTION 5: INTERNAL LINK SUGGESTION ENGINE
# =============================================================================

"""Internal link suggestion engine (v4).

Pipeline per spec:
1. Site content index already built (paragraph-level embeddings, SECTION 4)
2. For each paragraph in the article under review, find semantically similar
   paragraphs across the rest of the site (forward direction).
3. Layer anchor guide (priority + correct anchor text + cannibalization),
   GSC "topic DNA" validation, and Semrush volume context on top.
4. Bidirectional: also find existing pages whose paragraphs are similar to
   THIS article's paragraphs, i.e. who should link back to it.
5. TF-IDF fallback only for target pages that aren't in the anchor guide,
   used purely to derive a reasonable anchor text suggestion (not for matching).
"""


STOPWORDS_FOR_ANCHOR = {"the", "a", "an", "of", "and", "to", "for", "in", "on", "with"}

CITATION_PATTERN = re.compile(
    r"\bet al\b|\bPresented at\b|(New England Journal|Blood|Lancet|Nat Med|JAMA|J Clin Oncol)\s*,?\s*(19|20)\d{2}",
    re.IGNORECASE,
)


def _is_citation_paragraph(text: str) -> bool:
    """References-section entries aren't places to put internal links."""
    return bool(CITATION_PATTERN.search(text))


def _derive_anchor_from_h1_or_title(h1: str, title: str) -> str:
    """Fallback anchor text when a page isn't in the anchor guide.
    Strips site suffixes, cuts at the first colon (subtitle), and trims to
    ~60 chars at a word boundary so anchors read naturally."""
    base = h1 or title or ""
    base = re.sub(r"\s*\|.*$", "", base)  # strip " | The Cancer News" suffixes
    if ":" in base and len(base) > 60:
        base = base.split(":")[0]
    base = base.strip()
    if len(base) > 60:
        base = base[:60].rsplit(" ", 1)[0]
    return base.strip()


def _tfidf_keyword_overlap(paragraph: str, candidate_text: str) -> float:
    """Cheap fallback signal (not primary matching) for pages with no anchor
    guide entry — used only to sanity-check that the derived anchor text
    actually reflects shared vocabulary, per the 'TF-IDF fallback for pages
    not in the anchor guide' requirement.
    """
    try:
        vec = TfidfVectorizer(stop_words="english").fit([paragraph, candidate_text])
        matrix = vec.transform([paragraph, candidate_text])
        return float((matrix[0].multiply(matrix[1])).sum())
    except ValueError:
        return 0.0


def _gsc_validates(target_url: str, paragraph: str, gsc_df) -> dict:
    """Does the suggested target page actually rank for terms related to
    this paragraph's topic? Returns a small validation summary.

    Uses a stricter check: at least 2 distinct non-trivial words from a
    GSC query must appear in the paragraph, to avoid false positives from
    single generic matches like 'cancer'.
    """
    queries = gsc_queries_for_page(gsc_df, target_url)
    if not queries:
        return {"validated": False, "matching_queries": []}
    para_lower = paragraph.lower()
    matches = []
    for q in queries:
        words = [w for w in q.lower().split() if len(w) > 3]
        overlap = [w for w in words if w in para_lower]
        if len(overlap) >= 2:
            matches.append(q)
    return {"validated": bool(matches), "matching_queries": matches[:5], "all_queries": queries[:10]}


def build_forward_suggestions(article_paragraphs: list, para_index: dict,
                                anchor_df, gsc_df, cannibal_map: dict,
                                exclude_url: str = None, top_k: int = 3) -> list:
    """For each paragraph in the article being checked, suggest which
    existing pages it should link OUT to.
    """
    if not article_paragraphs:
        return []
    embeddings = embed_paragraphs(article_paragraphs)
    suggestions = []

    for i, (para_text, para_emb) in enumerate(zip(article_paragraphs, embeddings)):
        if len(para_text) < 80 or _is_citation_paragraph(para_text):
            continue  # headings, fragments, and reference entries aren't link spots
        matches = match_paragraph_against_index(para_emb, para_index, exclude_url=exclude_url, top_k=top_k)
        for m in matches:
            target_url = m["url"]
            anchor_entry = anchor_entry_for_url(anchor_df, target_url)
            if anchor_entry:
                anchor_text = anchor_entry.get("Primary Anchor Text") or _derive_anchor_from_h1_or_title(m["h1"], m["title"])
                priority = "anchor_guide"
                notes = anchor_entry.get("SEO Notes / Warnings", "")
            else:
                anchor_text = _derive_anchor_from_h1_or_title(m["h1"], m["title"])
                priority = "derived"
                notes = ""

            banned_anchors = anchor_bans_for_url(anchor_df, target_url)
            if banned_anchors and anchor_text and anchor_text.lower() in [b.lower() for b in banned_anchors]:
                # Fall back to an alternate anchor from the guide, or the derived one
                alt = (anchor_entry or {}).get("Alternate Anchor Texts", "")
                alt_options = [a.strip() for a in re.split(r"[\n,]", alt) if a.strip()]
                usable_alts = [a for a in alt_options if a.lower() not in [b.lower() for b in banned_anchors]]
                anchor_text = usable_alts[0] if usable_alts else _derive_anchor_from_h1_or_title(m["h1"], m["title"])

            gsc_check = _gsc_validates(target_url, para_text, gsc_df)
            cannibal_warning = cannibal_map.get(target_url)

            suggestions.append({
                "direction": "outbound",
                "source_paragraph_index": i,
                "source_paragraph_excerpt": para_text[:160],
                "target_url": target_url,
                "target_title": m["title"],
                "target_h1": m["h1"],
                "target_paragraph_excerpt": m["text"][:160],
                "recommended_anchor_text": anchor_text,
                "priority": priority,
                "similarity": round(m["similarity"], 3),
                "cannibalization_warning": cannibal_warning,
                "gsc_validated": gsc_check["validated"],
                "gsc_matching_queries": gsc_check.get("matching_queries", []),
                "anchor_guide_notes": notes,
            })
    # Deduplicate by TARGET URL — the SOP itself forbids duplicate link
    # targets in one article, so we recommend each target exactly once,
    # at its best-matching paragraph.
    seen = {}
    for s in suggestions:
        key = s["target_url"]
        if key not in seen or s["similarity"] > seen[key]["similarity"]:
            seen[key] = s
    suggestions = list(seen.values())
    suggestions.sort(key=lambda s: s["similarity"], reverse=True)
    return suggestions


def build_bidirectional_suggestions(article_paragraphs: list, article_h1: str,
                                      article_url: str, para_index: dict,
                                      anchor_df, gsc_df, top_k: int = 5) -> list:
    """Which EXISTING pages should link TO this article, and where.

    We embed this article's own paragraphs and search the site index for
    similar paragraphs — those source pages are candidates to add a link
    pointing at this article.
    """
    if not article_paragraphs:
        return []
    embeddings = embed_paragraphs(article_paragraphs)
    reverse_hits = []

    for i, (para_text, para_emb) in enumerate(zip(article_paragraphs, embeddings)):
        if len(para_text) < 80 or _is_citation_paragraph(para_text):
            continue
        matches = match_paragraph_against_index(para_emb, para_index, exclude_url=article_url, top_k=top_k)
        for m in matches:
            if _is_citation_paragraph(m["text"]):
                continue  # don't suggest inserting links into a References list
            reverse_hits.append({
                "article_paragraph_index": i,
                "article_paragraph_excerpt": para_text[:160],
                "source_page_url": m["url"],
                "source_page_title": m["title"],
                "existing_paragraph_where_link_fits": m["text"][:220],
                "similarity": round(m["similarity"], 3),
            })

    # Suggested anchor text pointing at THIS article
    anchor_entry = anchor_entry_for_url(anchor_df, article_url) if article_url else None
    if anchor_entry:
        this_article_anchor = anchor_entry.get("Primary Anchor Text") or article_h1
    else:
        # Derive a natural-length anchor from the H1 (full 90+ char titles
        # make unusable anchor text)
        this_article_anchor = _derive_anchor_from_h1_or_title(article_h1, "") or "this article"

    for hit in reverse_hits:
        hit["direction"] = "inbound"
        hit["recommended_anchor_text"] = this_article_anchor

    # Deduplicate by SOURCE PAGE — one actionable "add a link on page X"
    # recommendation per page, at its best-matching paragraph.
    seen = {}
    for h in reverse_hits:
        key = h["source_page_url"]
        if key not in seen or h["similarity"] > seen[key]["similarity"]:
            seen[key] = h
    reverse_hits = list(seen.values())
    reverse_hits.sort(key=lambda s: s["similarity"], reverse=True)
    return reverse_hits


def run_link_engine(article_text: str, article_h1: str, para_index: dict,
                     anchor_df, gsc_df, article_url: str = None,
                     top_k_forward: int = 3, top_k_bidirectional: int = 5,
                     max_outbound: int = 10, max_inbound: int = 10) -> dict:
    """Top-level entry point used by app.py. Returns forward + bidirectional
    suggestions plus the cannibalization map, ready to render.
    """
    cannibal_map = cannibalization_pairs(anchor_df)
    paragraphs = split_into_paragraphs(article_text)

    forward = build_forward_suggestions(
        paragraphs, para_index, anchor_df, gsc_df, cannibal_map,
        exclude_url=article_url, top_k=top_k_forward,
    )
    bidirectional = build_bidirectional_suggestions(
        paragraphs, article_h1, article_url, para_index, anchor_df, gsc_df,
        top_k=top_k_bidirectional,
    )
    return {
        "paragraph_count": len(paragraphs),
        "forward_suggestions": forward[:max_outbound],
        "bidirectional_suggestions": bidirectional[:max_inbound],
        "cannibalization_map": cannibal_map,
    }


# =============================================================================
# SECTION 6: SOP / KORAY / LINK-HEALTH CHECKS
# =============================================================================

"""All non-link-suggestion checks: SOP structure/meta rules, Koray semantic
SEO framework checks, and link health (broken links / redirects / speed).

Each check function returns a dict:
    {"check": name, "status": "pass" | "warn" | "fail", "message": str, "details": Any}
so app.py can render them uniformly.
"""


MAIN_DOMAIN = "binaytara.org"


def _result(check, status, message, details=None):
    return {"check": check, "status": status, "message": message, "details": details or {}}


# ---------------------------------------------------------------------------
# Org name check — applies to every text field, always run first
# ---------------------------------------------------------------------------

def check_org_name(all_text: str) -> dict:
    hits = find_org_name_violations(all_text)
    if hits:
        return _result("Org name", "fail",
                        f"Found 'Binaytara Foundation' {len(hits)}x — should be 'Binaytara'.",
                        {"occurrences": hits})
    return _result("Org name", "pass", "No 'Binaytara Foundation' references found.")


# ---------------------------------------------------------------------------
# SOP: slug
# ---------------------------------------------------------------------------

def check_slug(slug: str) -> dict:
    if not slug:
        return _result("Slug", "warn", "No slug provided to check.")
    issues = []
    if slug != slug.lower():
        issues.append("contains uppercase characters")
    if re.search(r"[^a-z0-9\-]", slug):
        issues.append("contains characters other than lowercase letters, numbers, hyphens")
    if "_" in slug:
        issues.append("uses underscores instead of hyphens")
    id_like_numbers = [n for n in re.findall(r"\d{4,}", slug) if not (len(n) == 4 and 1900 <= int(n) <= 2099)]
    if id_like_numbers:
        issues.append("looks like it contains an ID/date number")
    if len(slug) > 60:
        issues.append(f"long ({len(slug)} chars) — aim shorter")
    stop_hits = [w for w in slug.split("-") if w in {"the", "a", "an", "of", "and", "to", "for"}]
    if stop_hits:
        issues.append(f"contains stop words: {', '.join(stop_hits)}")
    if issues:
        return _result("Slug", "warn", "; ".join(issues), {"slug": slug})
    return _result("Slug", "pass", "Slug looks clean.", {"slug": slug})


# ---------------------------------------------------------------------------
# SOP: meta title / description
# ---------------------------------------------------------------------------

def check_meta_title(title: str) -> dict:
    if not title:
        return _result("Meta title", "warn", "No meta title provided.")
    length = len(title)
    if length < 50 or length > 60:
        return _result("Meta title", "warn" if 40 <= length <= 70 else "fail",
                        f"{length} chars — SOP target is 50-60.", {"title": title, "length": length})
    if title != title.title() and not any(c.isupper() for c in title):
        return _result("Meta title", "warn", "Doesn't appear to be title case.", {"title": title})
    return _result("Meta title", "pass", f"{length} chars — within 50-60 range.", {"title": title})


def check_meta_description(desc: str) -> dict:
    if not desc:
        return _result("Meta description", "warn", "No meta description provided.")
    length = len(desc)
    issues = []
    if length < 140 or length > 155:
        issues.append(f"{length} chars — SOP target is 140-155")
    if '"' in desc:
        issues.append("contains double quotes")
    if issues:
        return _result("Meta description", "warn", "; ".join(issues), {"description": desc, "length": length})
    return _result("Meta description", "pass", f"{length} chars, no double quotes.", {"description": desc})


# ---------------------------------------------------------------------------
# SOP: headings
# ---------------------------------------------------------------------------

def check_headings(headings: list) -> dict:
    """headings: list of (level:int, text:str) in document order."""
    if not headings:
        return _result("Headings", "warn", "No headings detected.")
    h1s = [h for lvl, h in headings if lvl == 1]
    issues = []
    if len(h1s) == 0:
        issues.append("no H1 found")
    elif len(h1s) > 1:
        issues.append(f"{len(h1s)} H1s found — should be exactly one")
    else:
        h1_len = len(h1s[0])
        if h1_len < 50 or h1_len > 70:
            issues.append(f"H1 is {h1_len} chars — SOP target is 50-70")

    prev_level = None
    for lvl, text in headings:
        if prev_level is not None and lvl > prev_level + 1:
            issues.append(f"heading level skips from H{prev_level} to H{lvl} ('{text[:40]}')")
        prev_level = lvl

    h2s = [h for lvl, h in headings if lvl == 2]
    question_h2s = [h for h in h2s if h.strip().endswith("?")]
    if h2s and not question_h2s:
        issues.append("no H2s are phrased as questions (Koray/semantic SEO pattern)")

    if issues:
        return _result("Headings", "warn", "; ".join(issues), {"headings": headings})
    return _result("Headings", "pass", "One H1 in range, hierarchy clean, question-format H2s present.",
                    {"headings": headings})


# ---------------------------------------------------------------------------
# SOP: internal / external links
# ---------------------------------------------------------------------------

GENERIC_ANCHORS = {"click here", "read more", "here", "this link", "learn more", "this page"}


def check_internal_links(article_html_or_text: str, links: list, first_para_text: str = "") -> dict:
    """links: list of dicts {href, anchor_text, is_internal}"""
    issues = []
    seen_targets = {}
    for link in links:
        if not link.get("is_internal"):
            continue
        href = link["href"]
        anchor = clean_whitespace(link.get("anchor_text", ""))
        if anchor.lower() in GENERIC_ANCHORS:
            issues.append(f"non-descriptive anchor text '{anchor}' -> {href}")
        if not href.startswith("http"):
            issues.append(f"relative URL should be absolute: {href}")
        if "utm_" in href.lower():
            issues.append(f"internal link has UTM parameters: {href}")
        if domain_of(href) and domain_of(href) != MAIN_DOMAIN and MAIN_DOMAIN in domain_of(href):
            issues.append(f"links to a subdomain instead of main domain: {href}")
        seen_targets[href] = seen_targets.get(href, 0) + 1
        if first_para_text and href in first_para_text:
            issues.append(f"link appears in the first paragraph: {href}")

    dupes = [h for h, c in seen_targets.items() if c > 1]
    if dupes:
        issues.append(f"duplicate internal link target(s): {', '.join(dupes)}")

    if not links:
        return _result("Internal links", "warn", "No internal links detected in the draft.")
    if issues:
        return _result("Internal links", "warn", "; ".join(issues), {"link_count": len(links)})
    return _result("Internal links", "pass", f"{len(links)} internal link(s), no issues found.")


def check_external_links(links: list) -> dict:
    external = [l for l in links if not l.get("is_internal")]
    if len(external) > 3:
        return _result("External links", "warn", f"{len(external)} external links — SOP caps at 0-3.")
    missing_target = [l for l in external if l.get("target") != "_blank"]
    if missing_target:
        return _result("External links", "warn",
                        f"{len(missing_target)} external link(s) missing target=\"_blank\".")
    return _result("External links", "pass", f"{len(external)} external link(s), within limit.")


# ---------------------------------------------------------------------------
# SOP: images
# ---------------------------------------------------------------------------

def check_images(images: list) -> dict:
    """images: list of dicts {src, alt}"""
    if not images:
        return _result("Images", "warn", "No images detected.")
    issues = []
    for img in images:
        alt = (img.get("alt") or "").strip()
        if not alt:
            issues.append(f"missing alt text: {img.get('src', '')[:60]}")
            continue
        if len(alt) > 125:
            issues.append(f"alt text over 125 chars: {img.get('src', '')[:60]}")
        if alt.lower().startswith("image of"):
            issues.append(f"alt text starts with 'Image of': {img.get('src', '')[:60]}")
        if not alt.rstrip().endswith("."):
            issues.append(f"alt text missing trailing period: {img.get('src', '')[:60]}")
    if issues:
        return _result("Images", "warn", "; ".join(issues), {"image_count": len(images)})
    return _result("Images", "pass", f"{len(images)} image(s), alt text clean.")


# ---------------------------------------------------------------------------
# SOP: content quality
# ---------------------------------------------------------------------------

def check_content_quality(paragraphs: list, full_text: str) -> dict:
    issues = []
    long_paras = [p for p in paragraphs if len(re.split(r'(?<=[.!?])\s+', p)) > 4]
    if long_paras:
        issues.append(f"{len(long_paras)} paragraph(s) longer than 4 sentences")

    has_faq = bool(re.search(r"\bFAQ\b|frequently asked questions", full_text, re.IGNORECASE))
    if not has_faq:
        issues.append("no FAQ section detected")

    stats = re.findall(r"\b\d+(?:\.\d+)?%|\b\d{2,}\b", full_text)
    if stats and not re.search(r"\bsource\b|\baccording to\b|\[\d+\]", full_text, re.IGNORECASE):
        issues.append("statistics present without a nearby source/citation reference")

    if issues:
        return _result("Content quality", "warn", "; ".join(issues))
    return _result("Content quality", "pass", "Paragraph length, FAQ, and sourcing look fine.")


# ---------------------------------------------------------------------------
# SOP: author / trust
# ---------------------------------------------------------------------------

def check_author_trust(full_text: str, byline: str = None, pub_date: str = None) -> dict:
    issues = []
    if not byline:
        byline_match = re.search(r"\bBy\s+[A-Z][a-zA-Z.\-]+(?:\s+[A-Z][a-zA-Z.\-]+){0,3}", full_text)
        if not byline_match:
            issues.append("no byline detected")
    if not pub_date:
        date_match = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            full_text)
        if not date_match:
            issues.append("no publication date detected")
    if issues:
        return _result("Author/trust", "warn", "; ".join(issues))
    return _result("Author/trust", "pass", "Byline and publication date present.")


# ---------------------------------------------------------------------------
# SOP: technical meta
# ---------------------------------------------------------------------------

def check_technical_meta(og_image: str = None, canonical: str = None,
                          robots: str = None, viewport: str = None) -> dict:
    issues = []
    if og_image and "localhost" in og_image.lower():
        issues.append(f"OG image points to localhost: {og_image}")
    if not og_image:
        issues.append("no OG image found")
    if not canonical:
        issues.append("no canonical URL found")
    if robots and "noindex" in robots.lower():
        issues.append("robots meta contains noindex")
    if not viewport:
        issues.append("no viewport meta tag found")
    if issues:
        return _result("Technical meta", "warn" if "noindex" not in "; ".join(issues) else "fail",
                        "; ".join(issues))
    return _result("Technical meta", "pass", "OG image, canonical, robots, and viewport all clean.")


# ---------------------------------------------------------------------------
# Koray framework checks
# ---------------------------------------------------------------------------

HEDGING_PHRASES = [
    "might", "could", "may", "possibly", "perhaps", "it is thought",
    "some believe", "in some cases", "it could be argued", "seems to",
]


def check_hedging_language(full_text: str) -> dict:
    lower = full_text.lower()
    hits = [p for p in HEDGING_PHRASES if p in lower]
    count = sum(lower.count(p) for p in hits)
    if count > 5:
        return _result("Hedging language (Koray)", "warn",
                        f"{count} hedging phrase occurrences — aim for more assertive, source-backed statements.",
                        {"phrases_found": hits})
    return _result("Hedging language (Koray)", "pass", f"{count} hedging phrase occurrence(s) — acceptable.")


def check_definition_patterns(full_text: str) -> dict:
    has_definition = bool(re.search(r"\b\w+\s+is\s+(a|an|the)\b", full_text)) or \
        bool(re.search(r"\brefers to\b|\bis defined as\b", full_text, re.IGNORECASE))
    if not has_definition:
        return _result("Definition patterns (Koray)", "warn",
                        "No clear 'X is a/an ...' or 'refers to' definition sentence detected near the top.")
    return _result("Definition patterns (Koray)", "pass", "Definitional sentence pattern detected.")


def check_title_vs_h1(meta_title: str, h1: str) -> dict:
    if not meta_title or not h1:
        return _result("Title vs H1 (Koray)", "warn", "Missing meta title or H1 to compare.")
    if meta_title.strip().lower() == h1.strip().lower():
        return _result("Title vs H1 (Koray)", "warn",
                        "Meta title and H1 are identical — Koray's framework recommends slight variation "
                        "so each targets a distinct but related query.")
    return _result("Title vs H1 (Koray)", "pass", "Meta title and H1 are differentiated.")


def check_thin_content(full_text: str) -> dict:
    word_count = len(full_text.split())
    if word_count < 300:
        return _result("Thin content (Koray)", "fail", f"{word_count} words — under the 300-word floor.")
    return _result("Thin content (Koray)", "pass", f"{word_count} words.")


def check_readability(full_text: str) -> dict:
    if not full_text.strip():
        return _result("Readability (Koray)", "warn", "No text to score.")
    try:
        score = textstat.flesch_reading_ease(full_text)
    except Exception:
        return _result("Readability (Koray)", "warn", "Could not compute Flesch score.")
    if score < 50:
        return _result("Readability (Koray)", "warn",
                        f"Flesch reading ease {score:.0f} — fairly difficult, consider shorter sentences.",
                        {"score": score})
    return _result("Readability (Koray)", "pass", f"Flesch reading ease {score:.0f}.", {"score": score})


# ---------------------------------------------------------------------------
# Link health (After Publishing mode)
# ---------------------------------------------------------------------------

def _check_one_link(url: str) -> dict:
    try:
        resp = requests.head(url, headers={"User-Agent": USER_AGENT}, timeout=10, allow_redirects=True)
        redirect_chain = len(resp.history)
        return {
            "url": url,
            "status_code": resp.status_code,
            "ok": resp.status_code < 400,
            "redirect_chain_length": redirect_chain,
            "response_time_ms": int(resp.elapsed.total_seconds() * 1000),
        }
    except requests.exceptions.RequestException as e:
        return {"url": url, "status_code": None, "ok": False, "error": str(e)}


def check_link_health(urls: list, max_workers: int = 10) -> dict:
    if not urls:
        return _result("Link health", "pass", "No links to check.")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for r in pool.map(_check_one_link, list(set(urls))):
            results.append(r)
    broken = [r for r in results if not r.get("ok")]
    slow = [r for r in results if r.get("response_time_ms", 0) > 2000]
    redirects = [r for r in results if r.get("redirect_chain_length", 0) > 1]

    issues = []
    if broken:
        issues.append(f"{len(broken)} broken link(s)")
    if slow:
        issues.append(f"{len(slow)} slow-responding link(s) (>2s)")
    if redirects:
        issues.append(f"{len(redirects)} link(s) with multi-hop redirect chains")

    status = "fail" if broken else ("warn" if (slow or redirects) else "pass")
    message = "; ".join(issues) if issues else "All links healthy."
    return _result("Link health", status, message, {"results": results})


# ---------------------------------------------------------------------------
# Runner: bundle all checks for one article
# ---------------------------------------------------------------------------

def run_all_sop_checks(payload: dict, is_draft: bool = False) -> list:
    """payload keys (all optional except full_text/paragraphs/headings):
    slug, meta_title, meta_description, headings, links, images, full_text,
    paragraphs, byline, pub_date, og_image, canonical, robots, viewport
    """
    checks = []
    all_text_for_org_check = " ".join(filter(None, [
        payload.get("full_text", ""), payload.get("meta_title", ""),
        payload.get("meta_description", ""), payload.get("slug", ""),
    ]))
    checks.append(check_org_name(all_text_for_org_check))
    checks.append(check_slug(payload.get("slug", "")))
    checks.append(check_meta_title(payload.get("meta_title", "")))
    checks.append(check_meta_description(payload.get("meta_description", "")))
    checks.append(check_headings(payload.get("headings", [])))
    if not is_draft:
        # Links, images, and technical meta only apply to live pages
        checks.append(check_internal_links(payload.get("full_text", ""), payload.get("links", []),
                                            first_para_text=(payload.get("paragraphs") or [""])[0]))
        checks.append(check_external_links(payload.get("links", [])))
        checks.append(check_images(payload.get("images", [])))
    checks.append(check_content_quality(payload.get("paragraphs", []), payload.get("full_text", "")))
    if not is_draft:
        checks.append(check_author_trust(payload.get("full_text", ""), payload.get("byline"), payload.get("pub_date")))
        checks.append(check_technical_meta(payload.get("og_image"), payload.get("canonical"),
                                            payload.get("robots"), payload.get("viewport")))
    checks.append(check_hedging_language(payload.get("full_text", "")))
    checks.append(check_definition_patterns(payload.get("full_text", "")))
    checks.append(check_title_vs_h1(payload.get("meta_title", ""), payload.get("h1", "")))
    checks.append(check_thin_content(payload.get("full_text", "")))
    checks.append(check_readability(payload.get("full_text", "")))
    return checks


# =============================================================================
# SECTION 7: STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="TCN Content SOP Validator", page_icon="✅", layout="wide")

# ---------------------------------------------------------------------------
# Password gate (optional — skipped when APP_PASSWORD isn't set)
# ---------------------------------------------------------------------------
def _get_app_password():
    pw = os.environ.get("APP_PASSWORD")
    if pw:
        return pw
    try:
        return st.secrets.get("APP_PASSWORD")
    except Exception:
        return None


APP_PASSWORD = _get_app_password()

if APP_PASSWORD:
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if not st.session_state.authed:
        st.title("🔒 TCN Content SOP Validator")
        pw = st.text_input("Password", type="password")
        if st.button("Enter"):
            if pw == APP_PASSWORD:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

st.title("✅ TCN Content SOP Validator (v4)")
st.caption(
    "Checks drafts and live pages against Binaytara's Content Publishing SOP, "
    "the Koray semantic-SEO framework, and site-wide internal linking rules."
)


# ---------------------------------------------------------------------------
# Load sheet data + build/reuse the site content index
# ---------------------------------------------------------------------------

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _cached_indexable_urls():
    all_pages = load_all_pages()
    indexable = get_indexable_urls(all_pages)
    anchor_df_local = load_anchor_guide()
    guide_urls = set(anchor_df_local["Page URL"].tolist())
    keep = []
    for u in indexable:
        if is_hub_or_nav_page(u):
            continue
        if is_content_article_url(u) or u in guide_urls:
            keep.append(u)
    return tuple(keep)


def get_or_build_index():
    if "para_index" in st.session_state:
        return st.session_state.para_index
    urls = _cached_indexable_urls()
    site_content = build_site_content_index(urls)
    para_index = build_paragraph_index(site_content)
    st.session_state.para_index = para_index
    st.session_state.site_content = site_content
    return para_index


with st.sidebar:
    st.header("Site index")
    try:
        anchor_df = load_anchor_guide()
        all_pages_df = load_all_pages()
        gsc_df = load_gsc_data()
        semrush_df = load_semrush_data()
        st.success(f"Anchor guide: {len(anchor_df)} rows")
        st.success(f"Crawl (All Page): {len(all_pages_df)} rows")
        st.success(f"GSC data: {len(gsc_df)} rows")
        st.success(f"Semrush data: {len(semrush_df)} rows")

        org_flags = org_name_flags_in_sheet(anchor_df)
        if org_flags:
            st.error(f"⚠️ {len(org_flags)} 'Binaytara Foundation' reference(s) found in the anchor guide sheet:")
            for f in org_flags:
                st.caption(f"**{f['url']}**\n\n{f['note']}")
    except Exception as e:
        st.error(f"Could not load Google Sheet data: {e}")
        st.stop()

    if st.button("Build / refresh site content index (paragraph embeddings)"):
        with st.spinner("Fetching article bodies and computing embeddings — first run takes a few minutes..."):
            st.session_state.pop("para_index", None)
            st.session_state.pop("site_content", None)
            get_or_build_index()
        st.success("Site content index ready.")

    if "para_index" in st.session_state:
        st.info(f"Index loaded: {len(st.session_state.para_index['meta'])} paragraphs "
                f"across {len(st.session_state.get('site_content', []))} pages.")
    else:
        st.warning("Index not built yet — click the button above before checking link suggestions.")


mode = st.radio("Mode", ["Before Publishing (draft)", "After Publishing (live URL)"], horizontal=True)

STATUS_ICON = {"pass": "🟢", "warn": "🟡", "fail": "🔴"}


def render_checks(checks: list):
    st.subheader("SOP / Koray / Technical checks")
    for c in checks:
        icon = STATUS_ICON.get(c["status"], "⚪")
        with st.expander(f"{icon} {c['check']} — {c['message']}", expanded=(c["status"] != "pass")):
            if c.get("details"):
                st.json(c["details"])


def render_link_suggestions(result: dict):
    st.subheader("🔗 Internal link suggestions")
    st.caption(f"{result['paragraph_count']} paragraph(s) analyzed against the site index.")

    fwd = result["forward_suggestions"]
    st.markdown(f"**Outbound — pages this article should link TO ({len(fwd)} suggestion(s))**")
    if not fwd:
        st.write("No suggestions above the similarity threshold.")
    for s in fwd[:20]:
        badge = "⭐ anchor guide" if s["priority"] == "anchor_guide" else "derived"
        gsc_badge = "✅ GSC-validated" if s["gsc_validated"] else "— no GSC overlap found"
        with st.expander(f"[{s['similarity']}] → {s['target_title'] or s['target_url']}  ({badge})"):
            st.write(f"**Recommended anchor text:** {s['recommended_anchor_text']}")
            st.write(f"**Target URL:** {s['target_url']}")
            st.write(f"**Place near article paragraph #{s['source_paragraph_index'] + 1}:** "
                     f"\"{s['source_paragraph_excerpt']}...\"")
            st.write(f"**Matched target paragraph:** \"{s['target_paragraph_excerpt']}...\"")
            st.write(f"**GSC validation:** {gsc_badge} {s['gsc_matching_queries']}")
            if s["cannibalization_warning"]:
                st.warning(f"⚠️ Cannibalization warning: {s['cannibalization_warning']}")
            if s["anchor_guide_notes"]:
                st.caption(f"Anchor guide notes: {s['anchor_guide_notes']}")

    bidi = result["bidirectional_suggestions"]
    st.markdown(f"**Inbound — existing pages that should link BACK to this article ({len(bidi)} suggestion(s))**")
    if not bidi:
        st.write("No suggestions above the similarity threshold.")
    for s in bidi[:20]:
        with st.expander(f"[{s['similarity']}] {s['source_page_url']}"):
            st.write(f"**Recommended anchor text:** {s['recommended_anchor_text']}")
            st.write(f"**This article's relevant paragraph:** \"{s['article_paragraph_excerpt']}...\"")
            st.write(f"**Existing paragraph on source page where the link fits:** "
                     f"\"{s['existing_paragraph_where_link_fits']}...\"")


# ---------------------------------------------------------------------------
# Mode: Before Publishing
# ---------------------------------------------------------------------------

if mode == "Before Publishing (draft)":
    st.markdown("Paste your draft below, or upload a `.docx` file. Optional front-matter lines "
                "(`Slug:`, `Meta Title:`, `Meta Description:`) at the top are auto-detected.")
    col1, col2 = st.columns(2)
    with col1:
        pasted = st.text_area("Paste draft text", height=350, placeholder=(
            "Slug: colorectal-cancer-screening-guidelines\n"
            "Meta Title: Colorectal Cancer Screening Guidelines 2026\n"
            "Meta Description: ...\n\n"
            "# What Are Colorectal Cancer Screening Guidelines?\n\n"
            "Colorectal cancer screening is ...\n"
        ))
    with col2:
        uploaded = st.file_uploader("...or upload a .docx", type=["docx"])

    if st.button("Run checks", type="primary"):
        if uploaded is not None:
            parsed = parse_docx_draft(uploaded.read())
        elif pasted.strip():
            parsed = parse_pasted_draft(pasted)
        else:
            st.warning("Paste some text or upload a .docx first.")
            st.stop()

        front = parsed["front_matter"]
        payload = {
            "slug": front.get("slug", ""),
            "meta_title": front.get("meta title", ""),
            "meta_description": front.get("meta description", ""),
            "headings": parsed["headings"],
            "links": [],   # drafts don't carry real <a> tags — link checks apply post-publish
            "images": [],
            "full_text": parsed["full_text"],
            "paragraphs": parsed["paragraphs"],
            "h1": parsed["h1"] or "",
        }
        checks = run_all_sop_checks(payload, is_draft=True)
        render_checks(checks)

        if "para_index" not in st.session_state:
            st.info("Build the site content index in the sidebar to get internal link suggestions.")
        else:
            result = run_link_engine(
                parsed["full_text"], parsed["h1"] or "", st.session_state.para_index,
                anchor_df, gsc_df, article_url=None,
            )
            render_link_suggestions(result)

# ---------------------------------------------------------------------------
# Mode: After Publishing
# ---------------------------------------------------------------------------

else:
    urls_input = st.text_area("Paste one or more live URLs (one per line)", height=120)
    if st.button("Fetch & run checks", type="primary"):
        urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
        if not urls:
            st.warning("Paste at least one URL.")
            st.stop()

        for url in urls:
            st.markdown(f"---\n## {url}")
            page = fetch_page(url)
            if not page.get("ok"):
                st.error(f"Could not fetch page: {page.get('error')}")
                continue

            import requests as _requests
            html = _requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15).text
            details = extract_page_details(html, url)
            paragraphs = split_into_paragraphs(details["full_text"])

            payload = {
                **details,
                "paragraphs": paragraphs,
                "slug": url.rstrip("/").split("/")[-1],
            }
            checks = run_all_sop_checks(payload)
            render_checks(checks)

            link_urls = [l["href"] for l in details["links"]]
            health = check_link_health(link_urls)
            with st.expander(f"{STATUS_ICON.get(health['status'])} Link health — {health['message']}"):
                st.json(health["details"])

            if "para_index" not in st.session_state:
                st.info("Build the site content index in the sidebar to get internal link suggestions.")
            else:
                result = run_link_engine(
                    details["full_text"], details.get("h1", ""), st.session_state.para_index,
                    anchor_df, gsc_df, article_url=url,
                )
                render_link_suggestions(result)
