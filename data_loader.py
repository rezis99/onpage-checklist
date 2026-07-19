"""Loads the four Google Sheet tabs that power the SOP Validator.

All tabs live in one spreadsheet, shared as "anyone with link can view",
and are pulled via the gviz CSV export endpoint (no API key / OAuth needed).
"""
import io
import re
import requests
import pandas as pd
import streamlit as st

from modules.utils import (
    SPREADSHEET_ID,
    TAB_ANCHOR_GUIDE,
    TAB_ALL_PAGE,
    TAB_GSC,
    TAB_SEMRUSH,
    sheet_csv_url,
    USER_AGENT,
)

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
