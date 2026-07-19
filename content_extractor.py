"""Content extraction:
1. Fetch <article> body text from live TCN/IJCCD pages (site content index).
2. Parse pasted text / uploaded .docx drafts for the "Before Publishing" mode,
   detecting structure from markdown headings, bold-as-headings, or Word styles.
"""
import re
import concurrent.futures
import requests
from bs4 import BeautifulSoup
import streamlit as st

from modules.utils import USER_AGENT, clean_whitespace

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
    from modules.utils import domain_of

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
    paragraphs_text = clean_whitespace(scope.get_text(" "))

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
        text = clean_whitespace(article.get_text(" ")) if article else ""
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

    full_text = "\n\n".join(paragraphs)
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

    full_text = "\n\n".join(paragraphs)
    h1_candidates = [h for lvl, h in headings if lvl == 1]

    return {
        "front_matter": front_matter,
        "headings": headings,
        "paragraphs": paragraphs,
        "full_text": full_text,
        "h1": h1_candidates[0] if h1_candidates else None,
    }
