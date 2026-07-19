"""
TCN Content SOP Validator v3
=============================
Two modes: Before Publishing (draft) / After Publishing (live URLs)

Data sources:
  - Google Sheet: Internal Linking Anchor Guide + Screaming Frog crawl data
  - Sitemap: fallback URL list

Checks: SOP rules + Koray Semantic Quality + Link Health
        + Smart Internal Link Suggestions (guide-first, TF-IDF fallback)
"""

import io, os, re, math, json, time, csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==============================================================
# CONFIG
# ==============================================================
st.set_page_config(page_title="TCN SOP Validator", page_icon="✅", layout="wide")

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
if APP_PASSWORD:
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if not st.session_state.authed:
        pw = st.text_input("Password", type="password")
        if pw:
            if pw == APP_PASSWORD:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("Wrong password.")
        st.stop()

st.title("✅ TCN Content SOP Validator")
st.caption("Check articles against Binaytara's Content Publishing SOP — before or after publishing.")

# ==============================================================
# CONSTANTS
# ==============================================================
BINAYTARA_DOMAIN = "binaytara.org"
SUBDOMAINS_TO_AVOID = ["conference.binaytara.org", "education.binaytara.org"]
SHEET_ID = "1T3Hf0gY96o4tPKJH1lvLw94eHxC-kco0CXIJgsDZTkk"
ANCHOR_TAB = "Internal Linking Anchor Texts"
CRAWL_TAB = "All Page"
HDRS = {"User-Agent": "Mozilla/5.0 (compatible; BinaytaraSopChecker/3.0)"}

SLUG_STOP_WORDS = {
    "the","and","a","an","in","of","for","to","on","at","by","with","is","are",
    "was","were","be","been","has","have","had","this","that","it","its","from",
    "or","but","not","what","how","when","where","who","which","your","our",
    "their","you","we","should","could","would","can","will","do","does",
}
BAD_ANCHOR_TEXTS = {
    "click here","read more","learn more","here","link",
    "this article","read this","this page","more info",
}
HEDGING_PHRASES = [
    "some studies suggest","it is believed","it may be","it might be",
    "could potentially","might help","may help","should consider",
    "it is thought","some experts believe","there is some evidence",
    "possibly","it is possible that","some research suggests",
    "it seems that","it appears that","anecdotal evidence",
]


# ==============================================================
# GOOGLE SHEET LOADERS (cached)
# ==============================================================
def _sheet_csv_url(tab_name):
    from urllib.parse import quote
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={quote(tab_name)}"


@st.cache_data(ttl=1800, show_spinner="Loading anchor guide from Google Sheet...")
def load_anchor_guide():
    """Load Internal Linking Anchor Texts tab → list of guide entries."""
    try:
        resp = requests.get(_sheet_csv_url(ANCHOR_TAB), timeout=15)
        reader = csv.DictReader(io.StringIO(resp.text))
        guide = []
        for row in reader:
            url = row.get("Page URL", "").strip()
            if not url.startswith("http"):
                continue
            primary = row.get("Primary Anchor Text", "").strip().lower()
            alt_raw = row.get("Alternate Anchor Texts", "").strip()
            alternates = [a.strip().lower() for a in alt_raw.replace("\n", "\n").split("\n") if a.strip()]
            notes = row.get("SEO Notes / Warnings", "").strip()
            name = row.get("Page Name", "").strip()
            vol = row.get("Top KW Volume", "").strip()

            has_cannibalization = "cannibalization" in notes.lower()
            do_not_anchors = []
            # Extract "DO NOT use" anchor warnings
            for m in re.findall(r'DO NOT use ["\u201c]([^"\u201d]+)["\u201d]', notes, re.I):
                do_not_anchors.append(m.lower())

            all_anchors = []
            if primary:
                all_anchors.append(primary)
            all_anchors.extend(alternates)

            guide.append({
                "url": url,
                "name": name,
                "primary_anchor": primary,
                "alternate_anchors": alternates,
                "all_anchors": all_anchors,
                "volume": vol,
                "notes": notes,
                "has_cannibalization": has_cannibalization,
                "do_not_anchors": do_not_anchors,
            })
        return guide
    except Exception as e:
        st.warning(f"Could not load anchor guide: {e}")
        return []


@st.cache_data(ttl=1800, show_spinner="Loading page data from Google Sheet...")
def load_crawl_data():
    """Load All Page tab → dict of url -> {title, h1, meta_desc, indexable}."""
    try:
        resp = requests.get(_sheet_csv_url(CRAWL_TAB), timeout=30)
        reader = csv.DictReader(io.StringIO(resp.text))
        pages = {}
        for row in reader:
            url = row.get("Address", "").strip()
            if not url.startswith("http"):
                continue
            status = row.get("Status Code", "").strip()
            indexability = row.get("Indexability", "").strip()
            title = row.get("Title 1", "").strip()
            h1 = row.get("H1-1", "").strip()
            meta_desc = row.get("Meta Description 1", "").strip()

            pages[url] = {
                "title": title,
                "h1": h1,
                "meta_desc": meta_desc,
                "indexable": indexability.lower() == "indexable",
                "status": status,
                "combined_text": f"{title} {h1} {meta_desc}".strip(),
            }
        return pages
    except Exception as e:
        st.warning(f"Could not load crawl data: {e}")
        return {}


# ==============================================================
# INTERNAL LINK SUGGESTION ENGINE
# ==============================================================
def split_sentences(text):
    """Split text into sentences, preserving order."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip() and len(s.strip()) > 15]


def suggest_internal_links(content_text, current_url, existing_link_hrefs):
    """Two-tier suggestion: guide-based first, then TF-IDF for remaining pages."""
    guide = load_anchor_guide()
    crawl = load_crawl_data()

    current_clean = current_url.rstrip("/").lower() if current_url else ""
    existing_clean = {u.rstrip("/").lower() for u in existing_link_hrefs}
    content_lower = content_text.lower()
    sentences = split_sentences(content_text)

    suggestions = []
    already_suggested_urls = set()

    # --- TIER 1: Guide-based matching ---
    for entry in guide:
        url_clean = entry["url"].rstrip("/").lower()
        if url_clean == current_clean or url_clean in existing_clean:
            continue

        # Check if page is indexable in crawl data
        crawl_info = crawl.get(entry["url"], {})
        if crawl_info and not crawl_info.get("indexable", True):
            continue

        # Search for anchor text matches in content
        best_anchor = None
        best_sentence = None
        best_sentence_idx = None

        for anchor in entry["all_anchors"]:
            if not anchor or len(anchor) < 3:
                continue
            if anchor in content_lower:
                # Find the best sentence containing this anchor
                for idx, sent in enumerate(sentences):
                    if anchor in sent.lower():
                        if best_anchor is None or len(anchor) > len(best_anchor):
                            best_anchor = anchor
                            best_sentence = sent
                            best_sentence_idx = idx
                        break

        if best_anchor:
            # Determine recommended anchor text
            if best_anchor == entry["primary_anchor"]:
                rec_anchor = entry["primary_anchor"]
                anchor_note = "Primary anchor text"
            else:
                rec_anchor = entry["primary_anchor"] or best_anchor
                anchor_note = f"Matched on: \"{best_anchor}\" → use primary: \"{entry['primary_anchor']}\""

            warning = ""
            if entry["has_cannibalization"]:
                warning = f"⚠️ CANNIBALIZATION: {entry['notes'][:200]}"
            if entry["do_not_anchors"]:
                warning += f" 🚫 DO NOT use as anchor: {', '.join(entry['do_not_anchors'])}"

            suggestions.append({
                "url": entry["url"],
                "page_name": entry["name"],
                "page_title": crawl_info.get("title", entry["name"]),
                "recommended_anchor": rec_anchor,
                "anchor_note": anchor_note,
                "matched_phrase": best_anchor,
                "place_near_sentence": best_sentence,
                "sentence_index": best_sentence_idx,
                "volume": entry["volume"],
                "warning": warning,
                "source": "Anchor Guide",
                "similarity": 1.0,
            })
            already_suggested_urls.add(url_clean)

    # --- TIER 2: TF-IDF for pages not in guide ---
    # Build corpus from indexable crawl pages not already suggested
    candidate_urls = []
    candidate_texts = []

    for url, info in crawl.items():
        url_clean = url.rstrip("/").lower()
        if not info.get("indexable", False):
            continue
        if url_clean == current_clean or url_clean in existing_clean:
            continue
        if url_clean in already_suggested_urls:
            continue
        combined = info.get("combined_text", "")
        if len(combined) < 10:
            continue
        candidate_urls.append(url)
        candidate_texts.append(combined)

    if candidate_texts and len(content_text) > 50:
        try:
            all_docs = candidate_texts + [content_text]
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=5000,
                ngram_range=(1, 2),
            )
            tfidf_matrix = vectorizer.fit_transform(all_docs)
            content_vec = tfidf_matrix[-1]
            candidate_vecs = tfidf_matrix[:-1]
            sims = cosine_similarity(content_vec, candidate_vecs).flatten()

            # Get top matches above threshold
            top_indices = np.argsort(sims)[::-1]
            for idx in top_indices[:15]:
                score = sims[idx]
                if score < 0.08:
                    break
                url = candidate_urls[idx]
                info = crawl.get(url, {})
                title = info.get("title", "")
                h1 = info.get("h1", "")

                # Find which sentence best matches this page
                best_sent, best_sent_idx = _find_best_sentence(
                    sentences, title, h1, info.get("meta_desc", "")
                )

                # Derive anchor text from H1 or title (shorter one, cleaned)
                anchor_source = h1 if h1 and len(h1) < len(title or h1 + "x") else title
                # Trim to a reasonable anchor length
                anchor_words = anchor_source.split()[:6]
                derived_anchor = " ".join(anchor_words).rstrip(" |–-:").strip()

                suggestions.append({
                    "url": url,
                    "page_name": h1 or title,
                    "page_title": title,
                    "recommended_anchor": derived_anchor.lower(),
                    "anchor_note": f"Derived from page H1/title (similarity: {score:.2f})",
                    "matched_phrase": "",
                    "place_near_sentence": best_sent,
                    "sentence_index": best_sent_idx,
                    "volume": "",
                    "warning": "",
                    "source": "TF-IDF",
                    "similarity": round(score, 3),
                })
        except Exception:
            pass

    # Sort: guide matches first (by volume desc), then TF-IDF (by similarity desc)
    def sort_key(s):
        is_guide = 1 if s["source"] == "Anchor Guide" else 0
        vol = 0
        try:
            vol = int(s["volume"].replace(",", ""))
        except (ValueError, AttributeError):
            pass
        return (is_guide, vol, s["similarity"])

    suggestions.sort(key=sort_key, reverse=True)
    return suggestions


def _find_best_sentence(sentences, title, h1, meta_desc):
    """Find which sentence in the article is most relevant to the target page."""
    if not sentences:
        return None, None

    # Build a small keyword set from target page
    keywords = set()
    for text in [title, h1, meta_desc]:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        keywords.update(w for w in words if w not in SLUG_STOP_WORDS)

    if not keywords:
        return None, None

    best_sent, best_idx, best_overlap = None, None, 0
    for idx, sent in enumerate(sentences):
        sent_words = set(re.findall(r'\b[a-z]{3,}\b', sent.lower()))
        overlap = len(sent_words & keywords)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sent = sent
            best_idx = idx

    if best_overlap >= 2:
        return best_sent, best_idx
    return None, None


# ==============================================================
# DISPLAY HELPERS
# ==============================================================
def result(status, section, rule, detail):
    return {"status": status, "section": section, "rule": rule, "detail": detail}


def display_results(results, link_suggestions=None):
    if not results:
        st.info("No checks to display.")
        return

    pc = sum(1 for r in results if r["status"] == "PASS")
    wc = sum(1 for r in results if r["status"] == "WARN")
    fc = sum(1 for r in results if r["status"] == "FAIL")
    total = pc + wc + fc

    if total > 0:
        score = round(pc / total * 100)
        sc = "🟢" if score >= 80 else ("🟡" if score >= 60 else "🔴")
        st.markdown(f"### {sc} SOP Score: {score}% &nbsp;&nbsp; ({pc} passed, {wc} warnings, {fc} failed)")
    st.markdown("---")

    sections_seen = []
    for r in results:
        if r["section"] not in sections_seen:
            sections_seen.append(r["section"])

    for sec in sections_seen:
        sr = [r for r in results if r["section"] == sec]
        fails = sum(1 for r in sr if r["status"] == "FAIL")
        warns = sum(1 for r in sr if r["status"] == "WARN")
        ic = "🔴" if fails else ("🟡" if warns else "🟢")

        with st.expander(f"{ic} {sec} ({fails} fail, {warns} warn)", expanded=(fails > 0)):
            for r in sr:
                icons = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}
                st.markdown(f"{icons.get(r['status'], 'ℹ️')} **{r['rule']}** — {r['detail']}")

    # --- Internal Link Suggestions ---
    if link_suggestions:
        guide_sug = [s for s in link_suggestions if s["source"] == "Anchor Guide"]
        tfidf_sug = [s for s in link_suggestions if s["source"] == "TF-IDF"]

        with st.expander(f"🔗 Internal Link Suggestions — {len(guide_sug)} from Guide, "
                         f"{len(tfidf_sug)} from TF-IDF", expanded=True):

            if guide_sug:
                st.markdown("#### 📘 From Anchor Guide (high confidence)")
                for s in guide_sug:
                    vol_str = f" — KW vol: {s['volume']}" if s['volume'] else ""
                    st.markdown(f"**[{s['page_name']}]({s['url']})**{vol_str}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🏷️ Use anchor text: **\"{s['recommended_anchor']}\"**")
                    st.caption(f"   {s['anchor_note']}")
                    if s['place_near_sentence']:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;📍 Place near: *\"{s['place_near_sentence'][:120]}...\"*")
                    if s['warning']:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{s['warning']}")
                    st.markdown("---")

            if tfidf_sug:
                st.markdown("#### 🔍 From Content Similarity (review before using)")
                st.caption("These pages aren't in the anchor guide yet — verify relevance and choose appropriate anchor text.")
                for s in tfidf_sug:
                    st.markdown(f"**[{s['page_name'][:80]}]({s['url']})** — similarity: {s['similarity']}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🏷️ Suggested anchor: **\"{s['recommended_anchor']}\"**")
                    if s['place_near_sentence']:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;📍 Place near: *\"{s['place_near_sentence'][:120]}...\"*")
                    st.markdown("---")


# ==============================================================
# SOP CHECK FUNCTIONS
# ==============================================================
def check_slug(slug, results):
    sec = "1. URL / Slug"
    if not slug:
        results.append(result("FAIL", sec, "Slug present", "No slug found."))
        return
    if slug != slug.lower():
        results.append(result("FAIL", sec, "All lowercase", f"Uppercase in: `{slug}`"))
    else:
        results.append(result("PASS", sec, "All lowercase", "Lowercase."))
    if "_" in slug:
        results.append(result("FAIL", sec, "Hyphens as separators", "Uses underscores."))
    else:
        results.append(result("PASS", sec, "Hyphens as separators", "Hyphens."))
    if re.search(r'[^a-z0-9\-]', slug):
        results.append(result("FAIL", sec, "No special characters", f"Found: {set(re.findall(r'[^a-z0-9-]', slug))}"))
    else:
        results.append(result("PASS", sec, "No special characters", "Clean."))
    parts = slug.split("-")
    if len(parts) > 8:
        results.append(result("WARN", sec, "Keep it short", f"{len(parts)} words."))
    else:
        results.append(result("PASS", sec, "Keep it short", f"{len(parts)} words."))


def check_meta_title(title, results):
    sec = "2. Meta Title"
    if not title or not title.strip():
        results.append(result("FAIL", sec, "Present", "No meta title."))
        return ""
    title = title.strip()
    l = len(title)
    if l < 50:
        results.append(result("WARN", sec, "Length (50–60)", f"{l} chars — under 50."))
    elif l <= 60:
        results.append(result("PASS", sec, "Length (50–60)", f"{l} chars."))
    else:
        results.append(result("FAIL", sec, "Length (50–60)", f"{l} chars — will truncate."))
    if len(title.split()) > 2 and title == title.lower():
        results.append(result("WARN", sec, "Title case", "All lowercase."))
    return title


def check_meta_description(desc, results):
    sec = "3. Meta Description"
    if not desc or not desc.strip():
        results.append(result("FAIL", sec, "Present", "Not found."))
        return
    l = len(desc.strip())
    if l < 140:
        results.append(result("WARN", sec, "Length (140–155)", f"{l} chars."))
    elif l <= 155:
        results.append(result("PASS", sec, "Length (140–155)", f"{l} chars."))
    else:
        results.append(result("FAIL", sec, "Length (140–155)", f"{l} chars — truncated."))
    if '"' in desc:
        results.append(result("FAIL", sec, "No double quotes", "Double quotes truncate snippets."))


def check_headings(headings, results):
    sec = "4. Headings"
    if not headings:
        results.append(result("FAIL", sec, "Present", "No headings."))
        return ""
    h1s = [(l, t) for l, t in headings if l == 1]
    if len(h1s) == 0:
        results.append(result("FAIL", sec, "Exactly one H1", "No H1."))
    elif len(h1s) == 1:
        results.append(result("PASS", sec, "Exactly one H1", f"\"{h1s[0][1][:60]}\""))
    else:
        results.append(result("FAIL", sec, "Exactly one H1", f"{len(h1s)} H1s."))

    if h1s:
        hl = len(h1s[0][1])
        if hl <= 70:
            results.append(result("PASS", sec, "H1 length (≤70)", f"{hl} chars."))
        else:
            results.append(result("FAIL", sec, "H1 length (≤70)", f"{hl} chars — most common mistake."))

    prev = 0
    for lvl, txt in headings:
        if lvl > prev + 1 and prev > 0:
            results.append(result("FAIL", sec, "Hierarchy", f"H{prev}→H{lvl} at \"{txt[:40]}\""))
            break
        prev = lvl
    else:
        results.append(result("PASS", sec, "Hierarchy", "No skipped levels."))

    h2s = [t for l, t in headings if l == 2]
    qh2 = [t for t in h2s if "?" in t]
    if h2s and not qh2:
        results.append(result("WARN", sec, "Question H2s", f"0/{len(h2s)} are questions."))
    elif h2s:
        results.append(result("PASS", sec, "Question H2s", f"{len(qh2)}/{len(h2s)}."))

    return h1s[0][1] if h1s else ""


def check_internal_links(links, results):
    sec = "5. Internal Links"
    internal = [l for l in links if l.get("is_internal")]
    if not internal:
        results.append(result("WARN", sec, "Present", "No internal links."))
        return
    results.append(result("INFO", sec, "Count", f"{len(internal)}."))
    fp = [l for l in internal if l.get("para_idx") == 0]
    if fp:
        results.append(result("FAIL", sec, "No first-paragraph links", f"{len(fp)} link(s)."))
    else:
        results.append(result("PASS", sec, "No first-paragraph links", "Clean."))
    tgts = [l["href"].rstrip("/").lower() for l in internal]
    dupes = {u: c for u, c in Counter(tgts).items() if c > 1}
    if dupes:
        results.append(result("FAIL", sec, "Link once only", f"Dupes: {list(dupes.keys())[:3]}"))
    else:
        results.append(result("PASS", sec, "Link once only", "No dupes."))
    bad = [l for l in internal if l.get("anchor", "").strip().lower() in BAD_ANCHOR_TEXTS]
    if bad:
        results.append(result("FAIL", sec, "Descriptive anchors", f"Generic: {[l['anchor'] for l in bad[:3]]}"))
    else:
        results.append(result("PASS", sec, "Descriptive anchors", "All descriptive."))
    rel = [l for l in internal if l["href"].startswith("/")]
    if rel:
        results.append(result("FAIL", sec, "Absolute URLs", f"{len(rel)} relative."))
    utm = [l for l in internal if re.search(r'[?&]utm_', l["href"].lower())]
    if utm:
        results.append(result("FAIL", sec, "No UTMs", f"{len(utm)} with tracking."))


def check_external_links(links, results):
    sec = "6. External Links"
    ext = [l for l in links if not l.get("is_internal")]
    if not ext:
        return
    if len(ext) <= 3:
        results.append(result("PASS", sec, "Count (0–3)", f"{len(ext)}."))
    else:
        results.append(result("WARN", sec, "Count (0–3)", f"{len(ext)}."))


def check_images(images, results):
    sec = "7. Images"
    if not images:
        return
    no_alt = [i for i in images if not i.get("alt", "").strip()]
    if no_alt:
        results.append(result("FAIL", sec, "Alt text", f"{len(no_alt)} missing."))
    else:
        results.append(result("PASS", sec, "Alt text", "All have alt."))
    for img in images:
        alt = img.get("alt", "").strip()
        if alt and len(alt) > 125:
            results.append(result("WARN", sec, "Alt ≤125 chars", f"{len(alt)} chars."))
        if alt and alt.lower().startswith(("image of", "photo of", "picture of")):
            results.append(result("WARN", sec, "No 'Image of'", f"\"{alt[:40]}\""))


def check_content_quality(text, headings, results):
    sec = "8. Content & AI"
    if len(text) < 100:
        return
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if paras:
        long = sum(1 for p in paras if len(re.findall(r'[.!?]+', p)) > 4)
        if long > len(paras) * 0.3:
            results.append(result("WARN", sec, "Short paragraphs", f"{long}/{len(paras)} too long."))
        else:
            results.append(result("PASS", sec, "Short paragraphs", "OK."))
    h2l = [t.lower() for l, t in headings if l == 2]
    if any("faq" in h or "frequently" in h for h in h2l):
        results.append(result("PASS", sec, "FAQ section", "Found."))
    stats = re.findall(r'\d+\.?\d*\s*%', text)
    if stats:
        results.append(result("PASS", sec, "Statistics", f"{len(stats)} figures."))
    else:
        results.append(result("WARN", sec, "Statistics", "None found."))


def check_semantic_quality(text, meta_title, h1_text, results):
    """Category C: Koray framework checks."""
    sec = "C. Semantic Quality"
    tl = text.lower()

    # Hedging
    found = [(p, tl.count(p)) for p in HEDGING_PHRASES if p in tl]
    if not found:
        results.append(result("PASS", sec, "Hedging language", "Declarative statements — good."))
    elif len(found) <= 2:
        results.append(result("WARN", sec, "Hedging language",
                              f"Minor: {', '.join(f'{p} ({c}x)' for p, c in found)}"))
    else:
        results.append(result("FAIL", sec, "Hedging language", f"{len(found)} patterns found."))

    # Definitions
    defs = sum(len(re.findall(p, text, re.I)) for p in
               [r'\b\w+\s+is\s+(?:a|an|the|defined as)\b', r'\brefers to\b', r'\bknown as\b'])
    if defs >= 2:
        results.append(result("PASS", sec, "Definitions", f"{defs} found."))
    else:
        results.append(result("WARN", sec, "Definitions", "Define key terms for AI extraction."))

    # Title vs H1
    if meta_title and h1_text:
        mt = re.sub(r'\s*\|.*$', '', meta_title).strip().lower()
        h1 = h1_text.strip().lower()
        if mt == h1:
            results.append(result("WARN", sec, "Title ≠ H1", "Identical — missed keyword variation."))
        else:
            results.append(result("PASS", sec, "Title ≠ H1", "Differently worded."))

    # Thin content
    wc = len(text.split())
    if wc < 300:
        results.append(result("FAIL", sec, "Thin content", f"{wc} words."))
    elif wc < 600:
        results.append(result("WARN", sec, "Content depth", f"{wc} words."))
    else:
        results.append(result("PASS", sec, "Content depth", f"{wc:,} words."))

    # Readability
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sents) > 3 and wc > 30:
        avg_sl = wc / len(sents)
        syls = sum(max(1, len(re.findall(r'[aeiouy]+', w.lower()))) for w in text.split())
        fre = 206.835 - 1.015 * avg_sl - 84.6 * (syls / wc)
        fre = max(0, min(100, fre))
        if fre >= 40:
            results.append(result("PASS", sec, "Readability", f"Flesch {fre:.0f} — accessible."))
        else:
            results.append(result("WARN", sec, "Readability", f"Flesch {fre:.0f} — very dense."))


def check_link_health(links, soup, results):
    sec = "D. Link Health"
    internal = [l for l in links if l.get("is_internal") and l["href"].startswith("http")]
    unique = list(set(l["href"].rstrip("/") for l in internal))[:15]
    if not unique:
        return
    broken = []
    for url in unique:
        try:
            r = requests.head(url, headers=HDRS, timeout=8, allow_redirects=True)
            if r.status_code >= 400:
                broken.append(f"`{url}` → {r.status_code}")
        except Exception:
            broken.append(f"`{url}` → timeout")
    if broken:
        results.append(result("FAIL", sec, "Broken links", f"{len(broken)}: {'; '.join(broken[:3])}"))
    else:
        results.append(result("PASS", sec, "Broken links", f"All {len(unique)} return 200."))


def check_technical_meta(soup, results):
    sec = "10. Technical Meta"
    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        if "localhost" in og["content"]:
            results.append(result("FAIL", sec, "OG image", f"Localhost: {og['content'][:50]}"))
        else:
            results.append(result("PASS", sec, "OG image", "Valid."))
    else:
        results.append(result("WARN", sec, "OG image", "Missing."))

    canon = soup.find("link", attrs={"rel": "canonical"})
    if canon and canon.get("href"):
        results.append(result("PASS", sec, "Canonical", f"{canon['href'][:60]}"))
    else:
        results.append(result("WARN", sec, "Canonical", "Missing — duplicate risk."))

    robots = soup.find("meta", attrs={"name": "robots"})
    if robots and "noindex" in (robots.get("content", "").lower()):
        results.append(result("FAIL", sec, "Robots", f"NOINDEX: \"{robots['content']}\""))

    if not soup.find("meta", attrs={"name": "viewport"}):
        results.append(result("WARN", sec, "Viewport", "Missing mobile viewport."))


def check_author(soup, results):
    sec = "9. Author & Trust"
    txt = soup.get_text().lower()
    if any(i in txt for i in ["written by", "authored by", "by dr.", "author:"]) or soup.find("meta", attrs={"name": "author"}):
        results.append(result("PASS", sec, "Author", "Detected."))
    else:
        results.append(result("WARN", sec, "Author", "No byline found."))
    if soup.find("meta", attrs={"property": "article:published_time"}) or soup.find("time"):
        results.append(result("PASS", sec, "Date", "Found."))
    else:
        results.append(result("WARN", sec, "Date", "No date detected."))


# ==============================================================
# PAGE ANALYZERS
# ==============================================================
def extract_links(article):
    links = []
    for i, p in enumerate(article.find_all("p")):
        for a in p.find_all("a", href=True):
            href = a["href"].strip()
            lp = urlparse(href)
            links.append({
                "href": href, "anchor": a.get_text().strip(), "para_idx": i,
                "is_internal": BINAYTARA_DOMAIN in (lp.netloc or "") or (not lp.netloc and href.startswith("/")),
                "is_subdomain": any(s in (lp.netloc or "") for s in SUBDOMAINS_TO_AVOID),
                "new_tab": a.get("target") == "_blank",
                "has_target_attr": a.get("target"),
            })
    return links


def analyze_live_url(url):
    results = []
    try:
        start = time.time()
        resp = requests.get(url, headers=HDRS, timeout=20, allow_redirects=True)
        rt = time.time() - start
        resp.raise_for_status()
    except Exception as e:
        results.append(result("FAIL", "Connection", "Fetch", str(e)[:100]))
        return results, []

    soup = BeautifulSoup(resp.text, "html.parser")

    if rt > 3:
        results.append(result("FAIL", "D. Link Health", "Response time", f"{rt:.1f}s"))
    elif rt > 1:
        results.append(result("WARN", "D. Link Health", "Response time", f"{rt:.1f}s"))
    if resp.history:
        results.append(result("WARN", "D. Link Health", "Redirects", f"{len(resp.history)} hop(s)."))

    parsed = urlparse(resp.url)
    slug = parsed.path.strip("/").split("/")[-1] if parsed.path.strip("/") else ""
    check_slug(slug, results)

    title_tag = soup.find("title")
    meta_title = check_meta_title(title_tag.get_text().strip() if title_tag else "", results)

    desc_tag = soup.find("meta", attrs={"name": "description"})
    check_meta_description(desc_tag["content"].strip() if desc_tag and desc_tag.get("content") else "", results)

    headings = []
    for lv in range(1, 7):
        for tag in soup.find_all(f"h{lv}"):
            t = tag.get_text().strip()
            if t:
                headings.append((lv, t))
    h1_text = check_headings(headings, results)

    article = (soup.find("article") or soup.find("main")
               or soup.find("div", class_=re.compile(r"content|article|post|entry", re.I))
               or soup.find("body") or soup)

    paragraphs = [p.get_text().strip() for p in article.find_all("p") if len(p.get_text().strip()) > 20]
    links = extract_links(article)
    check_internal_links(links, results)
    check_external_links(links, results)

    images = [{"alt": img.get("alt", "")} for img in article.find_all("img")
              if not ("logo" in img.get("src", "").lower() or "icon" in img.get("src", "").lower())]
    check_images(images, results)

    full_text = "\n\n".join(paragraphs)
    check_content_quality(full_text, headings, results)
    check_author(soup, results)
    check_technical_meta(soup, results)
    check_semantic_quality(full_text, meta_title, h1_text, results)
    check_link_health(links, soup, results)

    existing = [l["href"] for l in links if l.get("is_internal")]
    suggestions = suggest_internal_links(full_text, url, existing)

    return results, suggestions


def read_docx_text(uploaded_file):
    try:
        from docx import Document
        doc = Document(io.BytesIO(uploaded_file.read()))
        lines = []
        for p in doc.paragraphs:
            t = p.text.strip()
            s = p.style.name.lower() if p.style else ""
            if "heading 1" in s or "title" in s:
                lines.append(f"# {t}")
            elif "heading 2" in s:
                lines.append(f"## {t}")
            elif "heading 3" in s:
                lines.append(f"### {t}")
            elif t:
                lines.append(t)
            else:
                lines.append("")
        return "\n".join(lines)
    except Exception as e:
        st.error(f"Error reading .docx: {e}")
        return None


def parse_draft_text(raw_text):
    results = []
    lines = raw_text.split("\n")
    meta_title, meta_desc, slug, h1_text = "", "", "", ""
    headings, body_lines = [], []

    fps = {
        "meta_title": re.compile(r'^(?:meta\s*title|title\s*tag|seo\s*title)\s*[:=\-–—]\s*(.+)', re.I),
        "meta_desc": re.compile(r'^(?:meta\s*desc(?:ription)?|description)\s*[:=\-–—]\s*(.+)', re.I),
        "slug": re.compile(r'^(?:slug|url|permalink)\s*[:=\-–—]\s*(.+)', re.I),
    }

    for line in lines:
        s = line.strip()
        if not s:
            body_lines.append("")
            continue
        matched = False
        for field, pat in fps.items():
            m = pat.match(s)
            if m:
                v = m.group(1).strip().strip('"\'')
                if field == "meta_title": meta_title = v
                elif field == "meta_desc": meta_desc = v
                elif field == "slug": slug = v.strip("/").split("/")[-1]
                matched = True
                break
        if matched:
            continue
        md = re.match(r'^(#{1,6})\s+(.+)', s)
        if md:
            lv, txt = len(md.group(1)), md.group(2).strip()
            headings.append((lv, txt))
            if lv == 1 and not h1_text: h1_text = txt
            body_lines.append(s)
            continue
        is_bold = s.startswith("**") and s.endswith("**")
        clean = s.strip("*#_")
        if is_bold and len(clean) < 100 and not clean.endswith("."):
            if not headings:
                headings.append((1, clean)); h1_text = clean
            else:
                headings.append((2, clean))
        body_lines.append(s)

    if not headings:
        for l in lines:
            s = l.strip().strip("*#_")
            if s and len(s) < 150:
                headings.append((1, s)); h1_text = s; break

    if slug: check_slug(slug, results)
    else: results.append(result("INFO", "1. URL / Slug", "Not provided", "Add `Slug: your-slug`"))
    if meta_title: meta_title = check_meta_title(meta_title, results)
    else: results.append(result("INFO", "2. Meta Title", "Not provided", "Add `Meta Title: ...`"))
    if meta_desc: check_meta_description(meta_desc, results)
    check_headings(headings, results)

    body_text = "\n".join(body_lines)
    check_content_quality(body_text, headings, results)
    check_semantic_quality(body_text, meta_title, h1_text, results)

    existing = re.findall(r'https?://[^\s\)\]]+', body_text)
    suggestions = suggest_internal_links(body_text, "", existing)
    return results, suggestions


# ==============================================================
# MAIN UI
# ==============================================================
# Sidebar: data status
with st.sidebar:
    st.subheader("Data Sources")
    guide = load_anchor_guide()
    crawl = load_crawl_data()
    indexable = sum(1 for v in crawl.values() if v.get("indexable"))
    st.success(f"📘 Anchor Guide: {len(guide)} pages")
    st.success(f"📊 Crawl Data: {len(crawl)} total, {indexable} indexable")
    st.caption(f"Sheet: `{SHEET_ID[:12]}...`\nCached 30 min. Reload the app to refresh.")

mode = st.radio("Mode", ["Before Publishing (check a draft)", "After Publishing (check live URLs)"],
                horizontal=True)
st.markdown("---")

if mode == "Before Publishing (check a draft)":
    st.subheader("Check a draft before publishing")
    st.caption(
        "Paste your article or upload .docx. Tip: add `Slug:`, `Meta Title:`, `Meta Description:` at the top."
    )
    inp = st.radio("Input", ["Paste text", "Upload .docx"], horizontal=True)
    draft_text = None
    if inp == "Paste text":
        draft_text = st.text_area("Paste article draft", height=300)
    else:
        up = st.file_uploader("Upload .docx", type=["docx"])
        if up:
            draft_text = read_docx_text(up)
            if draft_text:
                with st.expander("Preview", expanded=False):
                    st.text(draft_text[:2000])

    if st.button("▶ Check Draft", type="primary") and draft_text:
        with st.spinner("Analyzing..."):
            res, sug = parse_draft_text(draft_text)
        display_results(res, sug)

elif mode == "After Publishing (check live URLs)":
    st.subheader("Check live pages against the SOP")
    url_text = st.text_area("Paste URLs (one per line)", height=150)

    if st.button("▶ Check URLs", type="primary"):
        urls = [u.strip() for u in url_text.strip().splitlines() if u.strip()]
        if not urls:
            st.error("Paste at least one URL.")
        else:
            for i, url in enumerate(urls):
                if not url.startswith("http"):
                    url = "https://" + url
                st.markdown(f"## Page {i+1}: `{url}`")
                with st.spinner(f"Checking {url}..."):
                    res, sug = analyze_live_url(url)
                display_results(res, sug)
                if i < len(urls) - 1:
                    st.markdown("---")
