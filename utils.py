"""Shared constants and small helpers for the SOP Validator."""
import re
import urllib.parse

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
