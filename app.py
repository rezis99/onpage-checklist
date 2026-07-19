"""
TCN Content SOP Validator v2
=============================
Two modes:
  - Before Publishing: paste draft text or upload a .docx
  - After Publishing:  paste live URLs to crawl and check

Checks: SOP rules + Semantic Content Quality (Koray Framework)
        + Link Health + Internal Link Suggestions from sitemap.
"""

import io
import os
import re
import math
import json
import time
from collections import Counter
from urllib.parse import urlparse, parse_qs

import streamlit as st
import requests
from bs4 import BeautifulSoup

# ==============================================================
# PAGE CONFIG
# ==============================================================
st.set_page_config(page_title="TCN SOP Validator", page_icon="✅", layout="wide")

# ==============================================================
# OPTIONAL PASSWORD GATE
# ==============================================================
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
SITEMAP_INDEX_URL = "https://binaytara.org/sitemap.xml"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BinaytaraSopChecker/2.0)"}

SLUG_STOP_WORDS = {
    "the", "and", "a", "an", "in", "of", "for", "to", "on", "at", "by",
    "with", "is", "are", "was", "were", "be", "been", "has", "have", "had",
    "this", "that", "it", "its", "from", "or", "but", "not", "what", "how",
    "when", "where", "who", "which", "your", "our", "their", "you", "we",
    "should", "could", "would", "can", "will", "do", "does",
}

BAD_ANCHOR_TEXTS = {
    "click here", "read more", "learn more", "here", "link",
    "this article", "read this", "this page", "more info",
}

HEDGING_PHRASES = [
    "some studies suggest", "it is believed", "it may be", "it might be",
    "could potentially", "might help", "may help", "should consider",
    "it is thought", "some experts believe", "there is some evidence",
    "possibly", "it is possible that", "some research suggests",
    "it seems that", "it appears that", "anecdotal evidence",
]

# ==============================================================
# RESULT HELPERS
# ==============================================================
def result(status, section, rule, detail):
    return {"status": status, "section": section, "rule": rule, "detail": detail}


def display_results(results, link_suggestions=None):
    if not results:
        st.info("No checks to display.")
        return

    pass_count = sum(1 for r in results if r["status"] == "PASS")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    total = pass_count + warn_count + fail_count

    if total > 0:
        score = round(pass_count / total * 100)
        if score >= 80:
            sc = "🟢"
        elif score >= 60:
            sc = "🟡"
        else:
            sc = "🔴"
        st.markdown(f"### {sc} SOP Score: {score}% &nbsp;&nbsp; "
                    f"({pass_count} passed, {warn_count} warnings, {fail_count} failed)")
    st.markdown("---")

    sections_seen = []
    for r in results:
        if r["section"] not in sections_seen:
            sections_seen.append(r["section"])

    for sec in sections_seen:
        sec_results = [r for r in results if r["section"] == sec]
        fails = sum(1 for r in sec_results if r["status"] == "FAIL")
        warns = sum(1 for r in sec_results if r["status"] == "WARN")

        if fails > 0:
            ic = "🔴"
        elif warns > 0:
            ic = "🟡"
        else:
            ic = "🟢"

        with st.expander(f"{ic} {sec} ({fails} fail, {warns} warn)", expanded=(fails > 0)):
            for r in sec_results:
                icons = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}
                st.markdown(f"{icons.get(r['status'], 'ℹ️')} **{r['rule']}** — {r['detail']}")

    # Internal link suggestions (separate section)
    if link_suggestions:
        with st.expander(f"🔗 Internal Link Suggestions ({len(link_suggestions)} found)", expanded=True):
            st.caption("Pages on binaytara.org that match topics in your content but aren't linked yet.")
            for sug in link_suggestions[:15]:
                st.markdown(f"- **\"{sug['match_phrase']}\"** in your text → "
                            f"[{sug['suggested_url']}]({sug['suggested_url']})  \n"
                            f"  *Slug keywords: {sug['slug_keywords']}*")
            if len(link_suggestions) > 15:
                st.caption(f"...and {len(link_suggestions) - 15} more.")


# ==============================================================
# SITEMAP LOADER (cached, lightweight — only fetches XML)
# ==============================================================
@st.cache_data(ttl=3600, show_spinner="Loading sitemap...")
def load_sitemap_index():
    """Fetch all URLs from binaytara.org sitemaps and build a topic index from slugs."""
    pages = []  # list of {url, section, slug, keywords}
    try:
        resp = requests.get(SITEMAP_INDEX_URL, headers=REQUEST_HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "xml")
        sitemap_urls = [loc.text.strip() for loc in soup.find_all("loc")]
    except Exception:
        return pages

    for sm_url in sitemap_urls:
        try:
            resp = requests.get(sm_url, headers=REQUEST_HEADERS, timeout=15)
            sm_soup = BeautifulSoup(resp.text, "xml")
            for loc in sm_soup.find_all("loc"):
                url = loc.text.strip()
                parsed = urlparse(url)
                path = parsed.path.strip("/")
                slug = path.split("/")[-1] if path else ""

                # Determine section
                if "/cancernews/article/" in url:
                    section = "Cancer News"
                elif "/journal/article/" in url:
                    section = "IJCCD"
                elif "/projects/conferences/" in url:
                    section = "Conferences"
                elif "/news/" in url:
                    section = "Blog/News"
                else:
                    section = "Main Site"

                # Extract keywords from slug
                slug_words = set(slug.split("-")) - SLUG_STOP_WORDS - {""}
                # Remove very short words and pure numbers
                slug_words = {w for w in slug_words if len(w) > 2 and not w.isdigit()}

                if slug_words and slug:
                    pages.append({
                        "url": url,
                        "section": section,
                        "slug": slug,
                        "keywords": slug_words,
                        # Build multi-word phrases from slug for better matching
                        "phrases": build_slug_phrases(slug),
                    })
        except Exception:
            continue

    return pages


def build_slug_phrases(slug):
    """Build 2-3 word phrases from slug for matching against content."""
    words = [w for w in slug.split("-") if w and len(w) > 2 and w not in SLUG_STOP_WORDS]
    phrases = set()
    # Full slug as phrase
    full = " ".join(words)
    if len(words) >= 2:
        phrases.add(full)
    # 2-word and 3-word sliding windows
    for n in [3, 2]:
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            phrases.add(phrase)
    return phrases


def suggest_internal_links(content_text, current_url, existing_link_hrefs):
    """Find sitemap pages whose slug topics appear in the content but aren't linked."""
    sitemap_pages = load_sitemap_index()
    if not sitemap_pages:
        return []

    content_lower = content_text.lower()
    current_url_clean = current_url.rstrip("/").lower() if current_url else ""
    existing_clean = {u.rstrip("/").lower() for u in existing_link_hrefs}

    suggestions = []
    seen_urls = set()

    for page in sitemap_pages:
        page_url_clean = page["url"].rstrip("/").lower()

        # Skip self and already-linked pages
        if page_url_clean == current_url_clean:
            continue
        if page_url_clean in existing_clean:
            continue
        if page_url_clean in seen_urls:
            continue

        # Check if any slug phrase appears in the content
        best_match = None
        best_len = 0
        for phrase in page["phrases"]:
            if len(phrase) < 6:  # skip very short phrases
                continue
            if phrase in content_lower:
                if len(phrase) > best_len:
                    best_match = phrase
                    best_len = len(phrase)

        if best_match:
            suggestions.append({
                "suggested_url": page["url"],
                "section": page["section"],
                "match_phrase": best_match,
                "slug_keywords": ", ".join(sorted(page["keywords"])),
                "match_length": best_len,
            })
            seen_urls.add(page_url_clean)

    # Sort by match length (longer = more specific = better suggestion)
    suggestions.sort(key=lambda x: x["match_length"], reverse=True)
    return suggestions


# ==============================================================
# SOP CHECK FUNCTIONS (from v1)
# ==============================================================
def check_slug(slug, results):
    sec = "1. URL / Slug"
    if not slug:
        results.append(result("FAIL", sec, "Slug present", "No slug found."))
        return

    if slug != slug.lower():
        results.append(result("FAIL", sec, "All lowercase", f"Slug has uppercase: `{slug}`"))
    else:
        results.append(result("PASS", sec, "All lowercase", "Slug is lowercase."))

    if "_" in slug:
        results.append(result("FAIL", sec, "Hyphens as separators", "Uses underscores — use hyphens."))
    else:
        results.append(result("PASS", sec, "Hyphens as separators", "Uses hyphens correctly."))

    if re.search(r'[^a-z0-9\-]', slug):
        bad = set(re.findall(r'[^a-z0-9\-]', slug))
        results.append(result("FAIL", sec, "No special characters", f"Contains: {', '.join(bad)}"))
    else:
        results.append(result("PASS", sec, "No special characters", "Clean."))

    parts = slug.split("-")
    if len(parts) > 8:
        results.append(result("WARN", sec, "Keep it short", f"{len(parts)} words — consider shortening."))
    else:
        results.append(result("PASS", sec, "Keep it short", f"{len(parts)} words."))

    if re.search(r'\b\d{4,}\b', slug) and not re.search(r'\b20\d{2}\b', slug):
        results.append(result("WARN", sec, "No numbers or IDs", "Contains long number — verify it's not a database ID."))
    else:
        results.append(result("PASS", sec, "No numbers or IDs", "No suspicious numbers."))

    found_stops = set(slug.split("-")) & SLUG_STOP_WORDS
    if len(found_stops) > 2:
        results.append(result("WARN", sec, "Remove filler words", f"Stop words: {', '.join(sorted(found_stops))}"))


def check_meta_title(title, results):
    sec = "2. Meta Title"
    if not title or not title.strip():
        results.append(result("FAIL", sec, "Meta title present", "No meta title found."))
        return
    title = title.strip()
    length = len(title)

    if length < 50:
        results.append(result("WARN", sec, "Length (50–60 chars)", f"{length} chars — under 50, consider expanding."))
    elif length <= 60:
        results.append(result("PASS", sec, "Length (50–60 chars)", f"{length} chars."))
    else:
        results.append(result("FAIL", sec, "Length (50–60 chars)", f"{length} chars — will be truncated. Cut to ≤60."))

    if len(title.split()) > 2 and title == title.lower():
        results.append(result("WARN", sec, "Title case", "All lowercase — SOP says use title case."))


def check_meta_description(desc, results):
    sec = "3. Meta Description"
    if not desc or not desc.strip():
        results.append(result("FAIL", sec, "Meta description present", "Not found."))
        return
    desc = desc.strip()
    length = len(desc)

    if length < 120:
        results.append(result("WARN", sec, "Length (140–155 chars)", f"{length} chars — under 120, looks incomplete."))
    elif length < 140:
        results.append(result("WARN", sec, "Length (140–155 chars)", f"{length} chars — aim for 140–155."))
    elif length <= 155:
        results.append(result("PASS", sec, "Length (140–155 chars)", f"{length} chars."))
    else:
        results.append(result("FAIL", sec, "Length (140–155 chars)", f"{length} chars — will be truncated."))

    if '"' in desc:
        results.append(result("FAIL", sec, "No double quotes", "Double quotes can truncate snippet."))
    else:
        results.append(result("PASS", sec, "No double quotes", "Clean."))


def check_headings(headings, results):
    sec = "4. Headings"
    if not headings:
        results.append(result("FAIL", sec, "Headings present", "No headings found."))
        return

    h1s = [(l, t) for l, t in headings if l == 1]
    if len(h1s) == 0:
        results.append(result("FAIL", sec, "Exactly one H1", "No H1 found."))
    elif len(h1s) == 1:
        results.append(result("PASS", sec, "Exactly one H1", f"H1: \"{h1s[0][1][:80]}\""))
    else:
        results.append(result("FAIL", sec, "Exactly one H1", f"Found {len(h1s)} H1s — must have exactly one."))

    if h1s:
        h1_len = len(h1s[0][1])
        if h1_len <= 60:
            results.append(result("PASS", sec, "H1 length (50–70)", f"{h1_len} chars — ideal."))
        elif h1_len <= 70:
            results.append(result("PASS", sec, "H1 length (50–70)", f"{h1_len} chars — within range."))
        else:
            results.append(result("FAIL", sec, "H1 length (50–70)",
                                  f"{h1_len} chars — the most common mistake per SOP. Cut to ≤70."))

    prev = 0
    for lvl, txt in headings:
        if lvl > prev + 1 and prev > 0:
            results.append(result("FAIL", sec, "Heading hierarchy",
                                  f"Jumped H{prev} → H{lvl} at \"{txt[:50]}\""))
            break
        prev = lvl
    else:
        results.append(result("PASS", sec, "Heading hierarchy", "No skipped levels."))

    h2s = [t for l, t in headings if l == 2]
    q_h2s = [t for t in h2s if "?" in t]
    if h2s and len(q_h2s) == 0:
        results.append(result("WARN", sec, "Question-format H2s",
                              f"None of {len(h2s)} H2s are questions — SOP recommends this for AI visibility."))
    elif h2s:
        results.append(result("PASS", sec, "Question-format H2s", f"{len(q_h2s)}/{len(h2s)} are questions."))

    return h1s[0][1] if h1s else ""


def check_internal_links(links, paragraphs, results):
    sec = "5. Internal Links"
    internal = [l for l in links if l.get("is_internal")]

    if not internal:
        results.append(result("WARN", sec, "Internal links present", "No internal links found."))
        return

    results.append(result("INFO", sec, "Count", f"{len(internal)} internal link(s)."))

    first_para = [l for l in internal if l.get("paragraph_index") == 0]
    if first_para:
        results.append(result("FAIL", sec, "No links in first paragraph",
                              f"{len(first_para)} link(s) in first paragraph."))
    else:
        results.append(result("PASS", sec, "No links in first paragraph", "Clean."))

    targets = [l["href"].rstrip("/").lower() for l in internal]
    dupes = {u: c for u, c in Counter(targets).items() if c > 1}
    if dupes:
        results.append(result("FAIL", sec, "Link each page only once",
                              f"Duplicates: {'; '.join(f'{u} ({c}x)' for u, c in list(dupes.items())[:3])}"))
    else:
        results.append(result("PASS", sec, "Link each page only once", "No duplicates."))

    bad = [l for l in internal if l.get("anchor_text", "").strip().lower() in BAD_ANCHOR_TEXTS]
    if bad:
        results.append(result("FAIL", sec, "Descriptive anchor text",
                              f"Generic anchors: {', '.join(set(l['anchor_text'] for l in bad[:3]))}"))
    else:
        results.append(result("PASS", sec, "Descriptive anchor text", "All anchors are descriptive."))

    relative = [l for l in internal if l["href"].startswith("/")]
    if relative:
        results.append(result("FAIL", sec, "Full absolute URLs", f"{len(relative)} use relative paths."))
    else:
        results.append(result("PASS", sec, "Full absolute URLs", "All absolute."))

    utm = [l for l in internal if re.search(r'[?&]utm_', l["href"].lower())]
    if utm:
        results.append(result("FAIL", sec, "No UTM parameters", f"{len(utm)} have tracking params."))
    else:
        results.append(result("PASS", sec, "No UTM parameters", "Clean."))

    subdomain = [l for l in internal if l.get("is_subdomain")]
    if subdomain:
        results.append(result("WARN", sec, "Prefer main domain",
                              f"{len(subdomain)} link(s) to subdomains — prefer main domain."))


def check_external_links(links, results):
    sec = "6. External Links"
    external = [l for l in links if not l.get("is_internal")]
    count = len(external)
    if count == 0:
        results.append(result("INFO", sec, "External links", "None found."))
        return

    if count <= 3:
        results.append(result("PASS", sec, "Count (0–3)", f"{count} — within guideline."))
    else:
        results.append(result("WARN", sec, "Count (0–3)", f"{count} — SOP recommends 0–3."))

    no_blank = [l for l in external if l.get("has_target_attr") is not None and not l.get("new_tab")]
    if no_blank:
        results.append(result("WARN", sec, "Open in new tab",
                              f"{len(no_blank)} don't use target=\"_blank\"."))


def check_images(images, results):
    sec = "7. Images"
    if not images:
        results.append(result("INFO", sec, "Images", "No content images found."))
        return

    no_alt = [i for i in images if not i.get("alt", "").strip()]
    if no_alt:
        results.append(result("FAIL", sec, "Alt text present", f"{len(no_alt)} image(s) missing alt text."))
    else:
        results.append(result("PASS", sec, "Alt text present", "All images have alt text."))

    for img in images:
        alt = img.get("alt", "").strip()
        if not alt:
            continue
        if len(alt) > 125:
            results.append(result("WARN", sec, "Alt ≤125 chars", f"{len(alt)} chars: \"{alt[:40]}...\""))
        if alt.lower().startswith(("image of", "photo of", "picture of")):
            results.append(result("WARN", sec, "No 'Image of' prefix", f"\"{alt[:40]}...\""))
        if not alt.endswith("."):
            results.append(result("WARN", sec, "Alt ends with period", f"\"{alt[:40]}\""))


# ==============================================================
# NEW: CATEGORY C — SEMANTIC CONTENT QUALITY (Koray Framework)
# ==============================================================
def check_hedging_language(text, results):
    """C-10: Flag vague/hedging language — Koray says use declarative statements."""
    sec = "C. Semantic Quality"
    text_lower = text.lower()
    found = []
    for phrase in HEDGING_PHRASES:
        count = text_lower.count(phrase)
        if count > 0:
            found.append(f"\"{phrase}\" ({count}x)")

    if len(found) == 0:
        results.append(result("PASS", sec, "Hedging language",
                              "No hedging phrases found — content uses declarative statements."))
    elif len(found) <= 2:
        results.append(result("WARN", sec, "Hedging language",
                              f"Minor hedging found: {', '.join(found)}. "
                              "Consider replacing with factual, declarative statements."))
    else:
        results.append(result("FAIL", sec, "Hedging language",
                              f"{len(found)} hedging patterns: {', '.join(found[:5])}. "
                              "Koray's framework says declarative statements improve information retrieval score."))


def check_definition_patterns(text, results):
    """C-11: Does the article define key terms?"""
    sec = "C. Semantic Quality"
    definition_patterns = [
        r'\b\w+\s+is\s+(?:a|an|the|defined as)\b',
        r'\brefers to\b',
        r'\bknown as\b',
        r'\bmeans\b',
        r'\bdefined as\b',
    ]
    count = sum(len(re.findall(p, text, re.I)) for p in definition_patterns)

    if count >= 3:
        results.append(result("PASS", sec, "Definition patterns",
                              f"{count} definition pattern(s) found — good for AI extraction & featured snippets."))
    elif count >= 1:
        results.append(result("PASS", sec, "Definition patterns",
                              f"{count} definition pattern(s) found."))
    else:
        results.append(result("WARN", sec, "Definition patterns",
                              "No clear definitions found. Define key medical terms the first time they appear "
                              "(SOP 10.3 + Koray framework — definitions are high-signal for AI extraction)."))


def check_title_h1_relationship(meta_title, h1_text, results):
    """C-12: Title and H1 should cover same topic but be worded differently."""
    sec = "C. Semantic Quality"
    if not meta_title or not h1_text:
        return

    # Clean both for comparison
    mt_clean = re.sub(r'\s*\|.*$', '', meta_title).strip().lower()
    h1_clean = h1_text.strip().lower()

    if mt_clean == h1_clean:
        results.append(result("WARN", sec, "Title vs H1",
                              "Meta title and H1 are identical — SOP says they should cover the same "
                              "topic but be worded differently (missed opportunity for keyword variations)."))
    else:
        # Check for reasonable overlap
        mt_words = set(mt_clean.split()) - SLUG_STOP_WORDS
        h1_words = set(h1_clean.split()) - SLUG_STOP_WORDS
        if mt_words and h1_words:
            overlap = len(mt_words & h1_words) / max(len(mt_words), len(h1_words))
            if overlap < 0.2:
                results.append(result("WARN", sec, "Title vs H1",
                                      f"Very low overlap between meta title and H1 ({overlap:.0%}) — "
                                      "they should cover the same topic."))
            else:
                results.append(result("PASS", sec, "Title vs H1",
                                      "Meta title and H1 are related but differently worded — good."))


def check_thin_content(text, results):
    """C-13: Flag pages under 300 words."""
    sec = "C. Semantic Quality"
    words = text.split()
    wc = len(words)

    if wc < 300:
        results.append(result("FAIL", sec, "Thin content",
                              f"Only {wc} words. Pages under 300 words rarely rank — "
                              "73% of TCN articles with zero traffic are thin."))
    elif wc < 600:
        results.append(result("WARN", sec, "Content depth",
                              f"{wc} words — consider expanding for better topical coverage."))
    else:
        results.append(result("PASS", sec, "Content depth", f"{wc:,} words."))


def check_readability(text, results):
    """C-14: Flesch-Kincaid readability approximation."""
    sec = "C. Semantic Quality"
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    if len(sentences) < 3 or len(words) < 30:
        return

    # Syllable count approximation
    def syllables(word):
        word = word.lower().strip(".,!?;:")
        if len(word) <= 3:
            return 1
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith('e'):
            count -= 1
        return max(1, count)

    total_syllables = sum(syllables(w) for w in words)
    avg_sentence_len = len(words) / len(sentences)
    avg_syllables = total_syllables / len(words)

    # Flesch Reading Ease
    fre = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables
    fre = max(0, min(100, fre))

    # Flesch-Kincaid Grade Level
    grade = 0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59
    grade = max(1, min(20, grade))

    if fre >= 60:
        results.append(result("PASS", sec, "Readability",
                              f"Flesch score {fre:.0f} (grade {grade:.1f}) — accessible to general audience. "
                              "SOP says 'plain language alongside medical terminology.'"))
    elif fre >= 40:
        results.append(result("PASS", sec, "Readability",
                              f"Flesch score {fre:.0f} (grade {grade:.1f}) — college-level, acceptable for medical content."))
    else:
        results.append(result("WARN", sec, "Readability",
                              f"Flesch score {fre:.0f} (grade {grade:.1f}) — very dense. Consider simpler sentences "
                              "and defining jargon. SOP 10.3: 'Plain language alongside medical terminology.'"))


# ==============================================================
# NEW: CATEGORY D — LINK HEALTH
# ==============================================================
def check_broken_internal_links(links, results):
    """D-15: Verify internal links return 200."""
    sec = "D. Link Health"
    internal = [l for l in links if l.get("is_internal") and l["href"].startswith("http")]

    if not internal:
        return

    unique_urls = list(set(l["href"].rstrip("/") for l in internal))[:20]  # cap at 20

    broken = []
    for url in unique_urls:
        try:
            r = requests.head(url, headers=REQUEST_HEADERS, timeout=8, allow_redirects=True)
            if r.status_code >= 400:
                broken.append(f"`{url}` → {r.status_code}")
        except Exception as e:
            broken.append(f"`{url}` → timeout/error")

    if broken:
        results.append(result("FAIL", sec, "Broken internal links",
                              f"{len(broken)} broken: {'; '.join(broken[:5])}"))
    else:
        results.append(result("PASS", sec, "Broken internal links",
                              f"All {len(unique_urls)} internal links return 200."))


def check_content_quality(text, headings, results):
    """SOP Sections 10 & 12 — Content quality + AI visibility."""
    sec = "8. Content & AI Visibility"
    if not text or len(text.strip()) < 100:
        return

    # Paragraph length
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if paragraphs:
        long = sum(1 for p in paragraphs if len(re.findall(r'[.!?]+', p)) > 4)
        if long > len(paragraphs) * 0.3:
            results.append(result("WARN", sec, "Short paragraphs (2–4 sentences)",
                                  f"{long}/{len(paragraphs)} paragraphs have 5+ sentences."))
        else:
            results.append(result("PASS", sec, "Short paragraphs", "Most are 2–4 sentences."))

    # FAQ section
    h2_lower = [t.lower() for l, t in headings if l == 2]
    if any("faq" in h or "frequently asked" in h for h in h2_lower):
        results.append(result("PASS", sec, "FAQ section", "Found — good for AI visibility."))
    else:
        results.append(result("INFO", sec, "FAQ section", "None detected — recommended for longer articles."))

    # Statistics
    stats = re.findall(r'\d+\.?\d*\s*%', text)
    if stats:
        results.append(result("PASS", sec, "Statistics present",
                              f"{len(stats)} percentage figure(s) — verify they include source & year."))
    else:
        results.append(result("WARN", sec, "Statistics with sources",
                              "No percentage figures found — articles with cited stats get more AI citations."))


def check_author_trust(soup, results):
    sec = "9. Author & Trust"
    page_text = soup.get_text().lower()
    indicators = ["written by", "authored by", "by dr.", "by dr ", "contributor", "author:"]
    author_meta = soup.find("meta", attrs={"name": "author"})

    if any(i in page_text for i in indicators) or author_meta:
        results.append(result("PASS", sec, "Author byline", "Author attribution detected."))
    else:
        results.append(result("WARN", sec, "Author byline", "No clear author byline detected."))

    date_meta = soup.find("meta", attrs={"property": "article:published_time"}) or soup.find("time")
    if date_meta:
        results.append(result("PASS", sec, "Publication date", "Date found."))
    else:
        results.append(result("WARN", sec, "Publication date", "No publication date detected."))


def check_technical_meta(soup, results):
    """OG/Twitter + canonical + robots."""
    sec = "10. Technical Meta"

    # OG image
    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        if "localhost" in og["content"]:
            results.append(result("FAIL", sec, "OG image", f"Points to localhost: {og['content'][:60]}"))
        else:
            results.append(result("PASS", sec, "OG image", "Present and valid."))
    else:
        results.append(result("WARN", sec, "OG image", "No og:image tag found."))

    # Canonical
    canon = soup.find("link", attrs={"rel": "canonical"})
    if canon and canon.get("href"):
        results.append(result("PASS", sec, "Canonical URL", f"Set to: {canon['href'][:80]}"))
    else:
        results.append(result("WARN", sec, "Canonical URL",
                              "No canonical tag found — risk of duplicate content."))

    # Robots
    robots = soup.find("meta", attrs={"name": "robots"})
    if robots and robots.get("content"):
        content = robots["content"].lower()
        if "noindex" in content:
            results.append(result("FAIL", sec, "Robots directive",
                                  f"Page has noindex — it won't appear in search results! Content: \"{robots['content']}\""))
        elif "nofollow" in content:
            results.append(result("WARN", sec, "Robots directive",
                                  f"Page has nofollow: \"{robots['content']}\""))
        else:
            results.append(result("PASS", sec, "Robots directive", "No blocking directives."))
    else:
        results.append(result("PASS", sec, "Robots directive", "No restrictive robots meta."))

    # Viewport
    viewport = soup.find("meta", attrs={"name": "viewport"})
    if viewport:
        results.append(result("PASS", sec, "Mobile viewport", "Viewport meta present."))
    else:
        results.append(result("WARN", sec, "Mobile viewport", "No viewport meta — may have mobile usability issues."))


# ==============================================================
# MAIN ANALYZERS
# ==============================================================
def analyze_live_url(url):
    results = []
    try:
        start = time.time()
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=20, allow_redirects=True)
        response_time = time.time() - start
        resp.raise_for_status()
    except Exception as e:
        results.append(result("FAIL", "Connection", "Fetch page", f"Could not fetch: {e}"))
        return results, []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Response time
    if response_time < 1:
        results.append(result("PASS", "D. Link Health", "Response time", f"{response_time:.2f}s"))
    elif response_time < 3:
        results.append(result("WARN", "D. Link Health", "Response time", f"{response_time:.2f}s — a bit slow."))
    else:
        results.append(result("FAIL", "D. Link Health", "Response time", f"{response_time:.2f}s — slow response."))

    # Redirect chain
    if resp.history:
        results.append(result("WARN", "D. Link Health", "Redirects",
                              f"{len(resp.history)} redirect(s) — each hop wastes crawl budget."))

    # Slug
    parsed = urlparse(resp.url)
    slug = parsed.path.strip("/").split("/")[-1] if parsed.path.strip("/") else ""
    check_slug(slug, results)

    # Meta
    title_tag = soup.find("title")
    meta_title = title_tag.get_text().strip() if title_tag else ""
    check_meta_title(meta_title, results)

    desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_desc = desc_tag["content"].strip() if desc_tag and desc_tag.get("content") else ""
    check_meta_description(meta_desc, results)

    # Headings
    headings = []
    for level in range(1, 7):
        for tag in soup.find_all(f"h{level}"):
            text = tag.get_text().strip()
            if text:
                headings.append((level, text))
    h1_text = check_headings(headings, results) or ""

    # Content area
    article = (soup.find("article") or soup.find("main")
               or soup.find("div", class_=re.compile(r"content|article|post|entry", re.I))
               or soup.find("body"))
    if not article:
        article = soup.find("body") or soup

    paragraphs = [p.get_text().strip() for p in article.find_all("p") if len(p.get_text().strip()) > 20]

    # Links
    all_links = []
    for i, p_tag in enumerate(article.find_all("p")):
        for a_tag in p_tag.find_all("a", href=True):
            href = a_tag["href"].strip()
            anchor = a_tag.get_text().strip()
            lp = urlparse(href)
            is_internal = BINAYTARA_DOMAIN in (lp.netloc or "") or (not lp.netloc and href.startswith("/"))
            is_subdomain = any(s in (lp.netloc or "") for s in SUBDOMAINS_TO_AVOID)
            all_links.append({
                "href": href, "anchor_text": anchor, "paragraph_index": i,
                "is_internal": is_internal, "is_subdomain": is_subdomain,
                "new_tab": a_tag.get("target") == "_blank",
                "has_target_attr": a_tag.get("target"),
            })

    check_internal_links(all_links, paragraphs, results)
    check_external_links(all_links, results)

    # Images
    images = []
    for img in article.find_all("img"):
        src = img.get("src", "")
        if img.get("width", "").isdigit() and int(img.get("width", "999")) < 50:
            continue
        if "logo" in src.lower() or "icon" in src.lower():
            continue
        images.append({"src": src, "alt": img.get("alt", "")})
    check_images(images, results)

    full_text = "\n\n".join(paragraphs)

    # SOP content quality
    check_content_quality(full_text, headings, results)
    check_author_trust(soup, results)
    check_technical_meta(soup, results)

    # Category C — Semantic quality
    check_hedging_language(full_text, results)
    check_definition_patterns(full_text, results)
    check_title_h1_relationship(meta_title, h1_text, results)
    check_thin_content(full_text, results)
    check_readability(full_text, results)

    # Category D — Link health
    check_broken_internal_links(all_links, results)

    # Internal link suggestions
    existing_hrefs = [l["href"] for l in all_links if l.get("is_internal")]
    suggestions = suggest_internal_links(full_text, url, existing_hrefs)

    return results, suggestions


def parse_draft_text(raw_text):
    results = []
    lines = raw_text.split("\n")
    meta_title, meta_desc, slug, h1_text = "", "", "", ""
    headings = []
    body_lines = []

    field_patterns = {
        "meta_title": re.compile(r'^(?:meta\s*title|title\s*tag|seo\s*title|page\s*title)\s*[:=\-–—]\s*(.+)', re.I),
        "meta_desc": re.compile(r'^(?:meta\s*desc(?:ription)?|seo\s*desc(?:ription)?|description)\s*[:=\-–—]\s*(.+)', re.I),
        "slug": re.compile(r'^(?:slug|url|permalink|url\s*slug)\s*[:=\-–—]\s*(.+)', re.I),
        "h1": re.compile(r'^(?:h1|heading\s*1|main\s*heading)\s*[:=\-–—]\s*(.+)', re.I),
    }

    for line in lines:
        stripped = line.strip()
        if not stripped:
            body_lines.append("")
            continue

        matched = False
        for field, pattern in field_patterns.items():
            m = pattern.match(stripped)
            if m:
                val = m.group(1).strip().strip('"\'')
                if field == "meta_title":
                    meta_title = val
                elif field == "meta_desc":
                    meta_desc = val
                elif field == "slug":
                    slug = val.strip("/").split("/")[-1]
                elif field == "h1":
                    headings.append((1, val))
                    h1_text = val
                matched = True
                break
        if matched:
            continue

        md = re.match(r'^(#{1,6})\s+(.+)', stripped)
        if md:
            level = len(md.group(1))
            text = md.group(2).strip()
            headings.append((level, text))
            if level == 1 and not h1_text:
                h1_text = text
            body_lines.append(stripped)
            continue

        is_bold = stripped.startswith("**") and stripped.endswith("**")
        clean = stripped.strip("*#_")
        if is_bold and len(clean) < 100 and not clean.endswith("."):
            if not headings:
                headings.append((1, clean))
                h1_text = clean
            else:
                headings.append((2, clean))
            body_lines.append(stripped)
            continue

        body_lines.append(stripped)

    if not headings:
        for line in lines:
            s = line.strip().strip("*#_")
            if s and len(s) < 150:
                headings.append((1, s))
                h1_text = s
                results.append(result("INFO", "Structure", "H1 guess",
                                      f"No heading markup — using first line: \"{s[:60]}\""))
                break

    if slug:
        check_slug(slug, results)
    else:
        results.append(result("INFO", "1. URL / Slug", "Not provided",
                              "Add `Slug: your-slug` at the top to check it."))

    if meta_title:
        check_meta_title(meta_title, results)
    else:
        results.append(result("INFO", "2. Meta Title", "Not provided",
                              "Add `Meta Title: Your Title` to check it."))

    if meta_desc:
        check_meta_description(meta_desc, results)
    else:
        results.append(result("INFO", "3. Meta Description", "Not provided",
                              "Add `Meta Description: Your desc` to check it."))

    check_headings(headings, results)

    body_text = "\n".join(body_lines)
    url_pattern = re.compile(r'https?://[^\s\)\]\>\"\']+')
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', body_text) if p.strip()]
    links = []
    for i, para in enumerate(paragraphs):
        for u in url_pattern.findall(para):
            p = urlparse(u)
            links.append({
                "href": u, "anchor_text": u, "paragraph_index": i,
                "is_internal": BINAYTARA_DOMAIN in (p.netloc or ""),
                "is_subdomain": any(s in (p.netloc or "") for s in SUBDOMAINS_TO_AVOID),
                "new_tab": None, "has_target_attr": None,
            })

    if links:
        check_internal_links(links, paragraphs, results)
        check_external_links(links, results)
    else:
        results.append(result("INFO", "5. Internal Links", "No links detected",
                              "Add internal links before publishing."))

    check_content_quality(body_text, headings, results)

    # Category C
    check_hedging_language(body_text, results)
    check_definition_patterns(body_text, results)
    if meta_title and h1_text:
        check_title_h1_relationship(meta_title, h1_text, results)
    check_thin_content(body_text, results)
    check_readability(body_text, results)

    # Internal link suggestions from sitemap
    existing_hrefs = [l["href"] for l in links if l.get("is_internal")]
    suggestions = suggest_internal_links(body_text, "", existing_hrefs)

    return results, suggestions


def read_docx_text(uploaded_file):
    try:
        from docx import Document
        doc = Document(io.BytesIO(uploaded_file.read()))
        lines = []
        for para in doc.paragraphs:
            text = para.text.strip()
            style = para.style.name.lower() if para.style else ""
            if "heading 1" in style or "title" in style:
                lines.append(f"# {text}")
            elif "heading 2" in style:
                lines.append(f"## {text}")
            elif "heading 3" in style:
                lines.append(f"### {text}")
            elif "heading 4" in style:
                lines.append(f"#### {text}")
            elif text:
                lines.append(text)
            else:
                lines.append("")
        return "\n".join(lines)
    except ImportError:
        st.error("python-docx not installed. Paste text instead.")
        return None
    except Exception as e:
        st.error(f"Error reading .docx: {e}")
        return None


# ==============================================================
# MAIN UI
# ==============================================================
mode = st.radio("Mode", ["Before Publishing (check a draft)", "After Publishing (check live URLs)"],
                horizontal=True)
st.markdown("---")

if mode == "Before Publishing (check a draft)":
    st.subheader("Check a draft before publishing")
    st.caption(
        "Paste your article or upload a .docx. Checks heading structure, content quality, "
        "link patterns, readability, hedging language, and suggests internal links from the sitemap.\n\n"
        "**Tip:** Add these at the top for extra checks:\n"
        "```\nSlug: your-planned-slug\n"
        "Meta Title: Your Title (Under 60 Chars)\n"
        "Meta Description: Your description 140-155 chars.\n```"
    )

    input_method = st.radio("Input", ["Paste text", "Upload .docx"], horizontal=True)
    draft_text = None

    if input_method == "Paste text":
        draft_text = st.text_area("Paste article draft", height=300,
                                  placeholder="Slug: pancreatic-cancer-awareness-month\n"
                                  "Meta Title: Pancreatic Cancer Awareness Month | The Cancer News\n\n"
                                  "# Pancreatic Cancer Awareness Month: Signs & Screening\n\n"
                                  "Pancreatic cancer remains one of the deadliest...\n\n"
                                  "## What Are the Early Signs?\n...")
    else:
        uploaded = st.file_uploader("Upload .docx", type=["docx"])
        if uploaded:
            draft_text = read_docx_text(uploaded)
            if draft_text:
                with st.expander("Preview extracted text", expanded=False):
                    st.text(draft_text[:2000] + ("..." if len(draft_text) > 2000 else ""))

    if st.button("▶ Check Draft", type="primary") and draft_text:
        with st.spinner("Analyzing draft + loading sitemap for link suggestions..."):
            res, sug = parse_draft_text(draft_text)
        display_results(res, sug)

elif mode == "After Publishing (check live URLs)":
    st.subheader("Check live pages against the SOP")
    st.caption("Paste URLs (one per line). Checks meta tags, headings, links, images, "
               "content quality, readability, hedging, broken links, and suggests internal links.")

    url_text = st.text_area("Paste URLs", height=150,
                            placeholder="https://binaytara.org/cancernews/article/pancreatic-cancer-awareness-month")

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
