"""All non-link-suggestion checks: SOP structure/meta rules, Koray semantic
SEO framework checks, and link health (broken links / redirects / speed).

Each check function returns a dict:
    {"check": name, "status": "pass" | "warn" | "fail", "message": str, "details": Any}
so app.py can render them uniformly.
"""
import re
import concurrent.futures
import requests
import textstat
from bs4 import BeautifulSoup

from modules.utils import USER_AGENT, clean_whitespace, find_org_name_violations, domain_of

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
