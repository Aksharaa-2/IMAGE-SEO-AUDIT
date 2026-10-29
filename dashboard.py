"""
dashboard.py — Interactive SEO dashboard for Image-Text Semantic Alignment.
Run with: streamlit run dashboard.py
"""
import json
import time
import traceback
from pathlib import Path

import streamlit as st
import pandas as pd

from src.extract import WebpageExtractor
from src.score import AlignmentScorer

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Image SEO Audit",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
    .app-title { font-size: 2.2rem; font-weight: 700; color: #1f2937; margin-bottom: 0.2rem; letter-spacing: -0.5px; }
    .app-subtitle { font-size: 1rem; color: #6b7280; margin-bottom: 2rem; }
    .score-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .score-label { font-size: 0.75rem; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
    .score-value { font-size: 2.4rem; font-weight: 700; color: #111827; line-height: 1.1; margin-top: 0.2rem; }
    .score-unit { font-size: 1rem; color: #9ca3af; font-weight: 500; }
    .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; }
    .badge-excellent { background: #d1fae5; color: #065f46; }
    .badge-good { background: #dbeafe; color: #1e40af; }
    .badge-moderate { background: #fef3c7; color: #92400e; }
    .badge-poor { background: #fee2e2; color: #991b1b; }
    .issue-block { background: #f9fafb; border-left: 3px solid #9ca3af; padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 0.7rem; }
    .issue-block-high { border-left-color: #dc2626; }
    .issue-block-medium { border-left-color: #f59e0b; }
    .issue-block-low { border-left-color: #10b981; }
    .issue-field { font-size: 0.7rem; font-weight: 700; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
    .issue-text { font-size: 0.92rem; color: #1f2937; margin-top: 0.15rem; }
    .suggestion-text { font-size: 0.88rem; color: #059669; font-style: italic; margin-top: 0.3rem; }
    .section-header { font-size: 1.15rem; font-weight: 700; color: #1f2937; padding-bottom: 0.5rem; border-bottom: 2px solid #e5e7eb; margin: 1.5rem 0 1rem 0; }
    .url-display { background: #f3f4f6; border-radius: 8px; padding: 0.6rem 1rem; font-family: monospace; font-size: 0.85rem; color: #374151; word-break: break-all; }
    .progress-track { height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden; margin-top: 0.4rem; }
    .progress-fill { height: 100%; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def score_color(score):
    if score >= 80: return "#10b981", "EXCELLENT"
    elif score >= 60: return "#3b82f6", "GOOD"
    elif score >= 40: return "#f59e0b", "MODERATE"
    else: return "#ef4444", "POOR"

def badge_class(gap_level):
    return f"badge-{gap_level.lower()}"

def score_bar(score, color):
    return f'<div class="progress-track"><div class="progress-fill" style="width:{score}%; background:{color};"></div></div>'

def priority_class(priority):
    return f"issue-block-{priority.lower()}"


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================
def generate_recommendations(item, score_result):
    recs = []
    alt = item.get("alt", "").strip()
    filename = item.get("filename", "")
    caption = item.get("caption", "").strip()

    if not alt:
        recs.append({"field": "alt", "issue": "Alt text is missing",
                     "suggestion": "Add alt text describing the visible content.",
                     "priority": "HIGH"})
    elif score_result["alignment_score"] < 60:
        recs.append({"field": "alt",
                     "issue": f"Alt text doesn't match image (score {score_result['alignment_score']}/100)",
                     "suggestion": "Rewrite alt text to describe what is actually visible.",
                     "priority": "HIGH"})

    generic = ["img", "image", "dsc", "photo", "untitled", "screenshot"]
    if any(g in filename.lower() for g in generic) or any(c.isdigit() for c in filename):
        recs.append({"field": "filename",
                     "issue": f"Filename '{filename}' is generic",
                     "suggestion": "Rename using descriptive keywords (e.g. 'red-car-highway.jpg').",
                     "priority": "MEDIUM"})

    if not caption:
        recs.append({"field": "caption", "issue": "Image has no caption",
                     "suggestion": "Add a caption describing the visible content.",
                     "priority": "MEDIUM"})

    if score_result["gap_level"] == "POOR":
        recs.append({"field": "content",
                     "issue": "Significant semantic gap between image and text",
                     "suggestion": "Update text to match the image, or replace the image.",
                     "priority": "HIGH"})

    return recs


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_scorer():
    return AlignmentScorer()


# ============================================================
# SESSION STATE
# ============================================================
if "report" not in st.session_state:
    st.session_state.report = None
if "scanned_url" not in st.session_state:
    st.session_state.scanned_url = None


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="app-title">Image SEO Audit</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Check whether image alt text, captions, and surrounding content actually match what appears in the image.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# INPUT SECTION
# ============================================================
col1, col2 = st.columns([4, 1])
with col1:
    url_input = st.text_input(
        "Website URL",
        placeholder="https://example.com/page",
        label_visibility="visible",
    )
with col2:
    st.write("")
    st.write("")
    run_button = st.button("Scan Website", type="primary", use_container_width=True)

with st.expander("Options"):
    max_images = st.slider("Maximum images to analyze", 1, 50, 15)


# ============================================================
# EXTRACTION AND SCORING
# ============================================================
if run_button and url_input.strip():
    url = url_input.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    progress = st.progress(0, text="Starting audit...")

    # Step 1: Extract
    progress.progress(10, text="Rendering page (this takes 5-15 seconds)...")
    try:
        extractor = WebpageExtractor()
        items = extractor.extract(url=url)
    except Exception as e:
        progress.empty()
        st.error(f"Could not load the webpage: {type(e).__name__}: {e}")
        with st.expander("Full error traceback"):
            st.code(traceback.format_exc(), language="python")
        st.info("**Try these working URLs:**\n- `https://en.wikipedia.org/wiki/Dog`\n- `https://en.wikipedia.org/wiki/Car`\n- `file:///E:/image_seo_project/data/sample_webpage.html`")
        st.stop()

    if not items:
        progress.empty()
        st.warning(
            "No usable images found on this page.\n\n"
            "**Possible reasons:**\n"
            "- The page blocks automated requests\n"
            "- All images are icons or tracking pixels smaller than 50px\n"
            "- Images require login to view\n\n"
            "**Try these working URLs:**\n"
            "- `https://en.wikipedia.org/wiki/Dog`\n"
            "- `https://en.wikipedia.org/wiki/Car`"
        )
        st.stop()

    items = items[:max_images]

    # Step 2: Load model
    progress.progress(40, text="Loading AI model...")
    scorer = load_scorer()

    # Step 3: Score each image
    progress.progress(50, text=f"Analyzing {len(items)} images...")
    report = []
    for i, item in enumerate(items, 1):
        combined_text = " ".join(filter(None, [
            item.get("alt", ""),
            item.get("caption", ""),
            item.get("surrounding_text", ""),
        ])).strip()

        result = scorer.score(item["image_path"], combined_text)
        recs = generate_recommendations(item, result)

        report.append({
            "image": item["filename"],
            "image_url": item["image_url"],
            "local_path": item["image_path"],
            "alt": item["alt"],
            "caption": item["caption"],
            "surrounding": item["surrounding_text"],
            **result,
            "recommendations": recs,
        })
        progress.progress(50 + int(45 * i / len(items)), text=f"Analyzing image {i}/{len(items)}...")

    progress.progress(100, text="Done.")
    time.sleep(0.3)
    progress.empty()

    st.session_state.report = report
    st.session_state.scanned_url = url

    # Save to disk
    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    out_file = report_dir / "seo_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


# ============================================================
# DISPLAY RESULTS
# ============================================================
if st.session_state.report:
    report = st.session_state.report
    url = st.session_state.scanned_url

    st.markdown("---")
    st.markdown(f'<div class="url-display">Scanned: {url}</div>', unsafe_allow_html=True)

    # ---------- SUMMARY METRICS ----------
    st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

    avg_score = sum(r["alignment_score"] for r in report) / len(report)
    poor_count = sum(1 for r in report if r["gap_level"] == "POOR")
    moderate_count = sum(1 for r in report if r["gap_level"] == "MODERATE")
    good_count = sum(1 for r in report if r["gap_level"] == "GOOD")
    excellent_count = sum(1 for r in report if r["gap_level"] == "EXCELLENT")
    total_recs = sum(len(r["recommendations"]) for r in report)

    c1, c2, c3, c4 = st.columns(4)
    color, _ = score_color(avg_score)

    with c1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Average Alignment</div>
            <div class="score-value">{avg_score:.1f}<span class="score-unit">/100</span></div>
            {score_bar(avg_score, color)}
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Images Analyzed</div>
            <div class="score-value">{len(report)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Issues Found</div>
            <div class="score-value">{poor_count + moderate_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Recommendations</div>
            <div class="score-value">{total_recs}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- PAGE HEALTH SUMMARY (Replaced Chart) ----------
    st.markdown('<div class="section-header">Page Health Summary</div>', unsafe_allow_html=True)

    total_imgs = len(report)
    pct_exc = (excellent_count / total_imgs) * 100
    pct_good = (good_count / total_imgs) * 100
    pct_mod = (moderate_count / total_imgs) * 100
    pct_poor = (poor_count / total_imgs) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🟢 Excellent", f"{excellent_count} images", f"{pct_exc:.0f}% of total")
    m2.metric("🔵 Good", f"{good_count} images", f"{pct_good:.0f}% of total")
    m3.metric("🟡 Moderate", f"{moderate_count} images", f"{pct_mod:.0f}% of total")
    m4.metric("🔴 Poor", f"{poor_count} images", f"{pct_poor:.0f}% of total")

    st.markdown("<br>", unsafe_allow_html=True)
    if poor_count > 0 or moderate_count > 0:
        st.info(
            f"**Action Required:** {poor_count + moderate_count} out of {total_imgs} images need attention. "
            "Scroll down to see the specific recommendations for the images flagged as **Poor** or **Moderate**."
        )
    else:
        st.success(
            "**Healthy Page:** All images on this page have good or excellent alignment between their text and visual content!"
        )

    # ---------- PER-IMAGE ANALYSIS ----------
    st.markdown('<div class="section-header">Image-by-Image Analysis</div>', unsafe_allow_html=True)

    for i, r in enumerate(report, 1):
        color, level_label = score_color(r["alignment_score"])
        badge = badge_class(r["gap_level"])

        with st.expander(
            f"{i}. {r['image'][:70]} — {r['alignment_score']}/100 ({level_label})",
            expanded=(r["gap_level"] in ("POOR", "MODERATE")),
        ):
            img_col, info_col = st.columns([1, 2])

            with img_col:
                try:
                    st.image(r["local_path"], width=220)
                except Exception:
                    st.caption("Image could not be displayed")

            with info_col:
                st.markdown(f'<div style="margin-bottom:0.8rem;"><span class="badge {badge}">{r["gap_level"]}</span></div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="margin-bottom:0.6rem;">
                    <div class="score-label">Alignment Score</div>
                    <div style="display:flex;align-items:baseline;gap:0.4rem;">
                        <span style="font-size:1.8rem;font-weight:700;color:{color};">{r['alignment_score']}</span>
                        <span style="color:#9ca3af;font-size:0.9rem;">/ 100</span>
                    </div>
                    {score_bar(r['alignment_score'], color)}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"**Match Probability:** {r['match_probability']*100:.1f}%")
                st.markdown(f"**Verdict:** {'✓ Image and text match' if r['prediction']=='MATCH' else '✗ Image and text do not match'}")

            st.markdown("**Alt Text:**")
            st.code(r["alt"] if r["alt"] else "(none)", language=None)

            if r["caption"]:
                st.markdown("**Caption:**")
                st.markdown(f"*{r['caption']}*")

            st.markdown("---")
            st.markdown("**Recommendations:**")

            if not r["recommendations"]:
                st.success("No issues found. This image is well-optimized.")
            else:
                for rec in r["recommendations"]:
                    st.markdown(f"""
                    <div class="issue-block {priority_class(rec['priority'])}">
                        <div class="issue-field">{rec['field']} · {rec['priority']} Priority</div>
                        <div class="issue-text">{rec['issue']}</div>
                        <div class="suggestion-text">→ {rec['suggestion']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ---------- DOWNLOAD REPORT ----------
    st.markdown("---")
    st.markdown('<div class="section-header">Download Report</div>', unsafe_allow_html=True)

    report_json = json.dumps(report, indent=2, default=str)
    report_csv = pd.DataFrame([
        {
            "image": r["image"],
            "alt_text": r["alt"],
            "alignment_score": r["alignment_score"],
            "gap_level": r["gap_level"],
            "verdict": r["prediction"],
            "num_recommendations": len(r["recommendations"]),
        }
        for r in report
    ]).to_csv(index=False)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("Download JSON Report", report_json, "seo_report.json", "application/json", use_container_width=True)
    with dl2:
        st.download_button("Download CSV Summary", report_csv, "seo_report.csv", "text/csv", use_container_width=True)

else:
    st.info("Enter a URL above and click **Scan Website** to begin the audit.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#9ca3af;font-size:0.8rem;padding:1rem 0;">'
    'Image SEO Audit · Detects mismatches between image content and alt text using CLIP embeddings'
    '</div>',
    unsafe_allow_html=True,
)