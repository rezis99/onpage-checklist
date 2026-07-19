import os
import streamlit as st

from modules import data_loader, content_extractor, embeddings, link_engine, sop_checks
from modules.utils import find_org_name_violations

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
    all_pages = data_loader.load_all_pages()
    return tuple(data_loader.get_indexable_urls(all_pages))


def get_or_build_index():
    if "para_index" in st.session_state:
        return st.session_state.para_index
    urls = _cached_indexable_urls()
    site_content = content_extractor.build_site_content_index(urls)
    para_index = embeddings.build_paragraph_index(site_content)
    st.session_state.para_index = para_index
    st.session_state.site_content = site_content
    return para_index


with st.sidebar:
    st.header("Site index")
    try:
        anchor_df = data_loader.load_anchor_guide()
        all_pages_df = data_loader.load_all_pages()
        gsc_df = data_loader.load_gsc_data()
        semrush_df = data_loader.load_semrush_data()
        st.success(f"Anchor guide: {len(anchor_df)} rows")
        st.success(f"Crawl (All Page): {len(all_pages_df)} rows")
        st.success(f"GSC data: {len(gsc_df)} rows")
        st.success(f"Semrush data: {len(semrush_df)} rows")

        org_flags = data_loader.org_name_flags_in_sheet(anchor_df)
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
            parsed = content_extractor.parse_docx_draft(uploaded.read())
        elif pasted.strip():
            parsed = content_extractor.parse_pasted_draft(pasted)
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
        checks = sop_checks.run_all_sop_checks(payload)
        render_checks(checks)

        if "para_index" not in st.session_state:
            st.info("Build the site content index in the sidebar to get internal link suggestions.")
        else:
            result = link_engine.run_link_engine(
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
            page = content_extractor.fetch_page(url)
            if not page.get("ok"):
                st.error(f"Could not fetch page: {page.get('error')}")
                continue

            import requests as _requests
            html = _requests.get(url, headers={"User-Agent": content_extractor.USER_AGENT}, timeout=15).text
            details = content_extractor.extract_page_details(html, url)
            paragraphs = [p for p in details["full_text"].split(". ") if len(p) > 25]

            payload = {
                **details,
                "paragraphs": paragraphs,
                "slug": url.rstrip("/").split("/")[-1],
            }
            checks = sop_checks.run_all_sop_checks(payload)
            render_checks(checks)

            link_urls = [l["href"] for l in details["links"]]
            health = sop_checks.check_link_health(link_urls)
            with st.expander(f"{STATUS_ICON.get(health['status'])} Link health — {health['message']}"):
                st.json(health["details"])

            if "para_index" not in st.session_state:
                st.info("Build the site content index in the sidebar to get internal link suggestions.")
            else:
                result = link_engine.run_link_engine(
                    details["full_text"], details.get("h1", ""), st.session_state.para_index,
                    anchor_df, gsc_df, article_url=url,
                )
                render_link_suggestions(result)
