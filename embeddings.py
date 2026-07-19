"""Embedding model + paragraph-level semantic matching.

Uses all-MiniLM-L6-v2 (22MB, free, runs fine on Streamlit Community Cloud's
free tier) to embed paragraphs and match them by cosine similarity — this
replaces the old TF-IDF-on-title/H1 approach with real content-level matching.
"""
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from modules.utils import split_into_paragraphs

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
        paras = split_into_paragraphs(page["text"])
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
                                    min_similarity: float = 0.45) -> list:
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
        results.append({**m, "similarity": score})
    return results
