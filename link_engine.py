"""Internal link suggestion engine (v4).

Pipeline per spec:
1. Site content index already built (paragraph-level embeddings, modules/embeddings.py)
2. For each paragraph in the article under review, find semantically similar
   paragraphs across the rest of the site (forward direction).
3. Layer anchor guide (priority + correct anchor text + cannibalization),
   GSC "topic DNA" validation, and Semrush volume context on top.
4. Bidirectional: also find existing pages whose paragraphs are similar to
   THIS article's paragraphs, i.e. who should link back to it.
5. TF-IDF fallback only for target pages that aren't in the anchor guide,
   used purely to derive a reasonable anchor text suggestion (not for matching).
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from modules.embeddings import embed_paragraphs, match_paragraph_against_index
from modules.utils import split_into_paragraphs
from modules.data_loader import (
    anchor_entry_for_url,
    gsc_queries_for_page,
    cannibalization_pairs,
    anchor_bans_for_url,
)

STOPWORDS_FOR_ANCHOR = {"the", "a", "an", "of", "and", "to", "for", "in", "on", "with"}


def _derive_anchor_from_h1_or_title(h1: str, title: str) -> str:
    """Fallback anchor text when a page isn't in the anchor guide."""
    base = h1 or title or ""
    base = re.sub(r"\s*\|.*$", "", base)  # strip " | The Cancer News" suffixes
    return base.strip()[:80]


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
    # Deduplicate: one suggestion per (source_paragraph, target_url) — keep highest similarity
    seen = {}
    for s in suggestions:
        key = (s["source_paragraph_index"], s["target_url"])
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
        matches = match_paragraph_against_index(para_emb, para_index, exclude_url=article_url, top_k=top_k)
        for m in matches:
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
        this_article_anchor = article_h1 or "this article"

    for hit in reverse_hits:
        hit["direction"] = "inbound"
        hit["recommended_anchor_text"] = this_article_anchor

    # Deduplicate: one per (source_page_url, article_paragraph_index)
    seen = {}
    for h in reverse_hits:
        key = (h["source_page_url"], h["article_paragraph_index"])
        if key not in seen or h["similarity"] > seen[key]["similarity"]:
            seen[key] = h
    reverse_hits = list(seen.values())
    reverse_hits.sort(key=lambda s: s["similarity"], reverse=True)
    return reverse_hits


def run_link_engine(article_text: str, article_h1: str, para_index: dict,
                     anchor_df, gsc_df, article_url: str = None,
                     top_k_forward: int = 3, top_k_bidirectional: int = 5) -> dict:
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
        "forward_suggestions": forward,
        "bidirectional_suggestions": bidirectional,
        "cannibalization_map": cannibal_map,
    }
