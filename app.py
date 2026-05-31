import os
import sys
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.image_metrics import analyze_image
from utils.qc_rules import assign_qc_status, THRESHOLDS
from utils.ml_demo import train_and_evaluate, make_feature_importance_chart, FEATURE_COLS

st.set_page_config(
    page_title="BioVision QC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("BioVision QC")
    st.caption("v1.0 | Research Use Only")
    st.markdown("---")

    st.subheader("QC Thresholds")
    st.caption(
        "Thresholds shown here are general defaults. They should be calibrated "
        "for your specific assay, stain, and imaging setup before use."
    )

    blur_fail = st.slider(
        "Blur Score - Fail below",
        min_value=0, max_value=300, value=int(THRESHOLDS["blur_fail"]), step=5,
        help="Laplacian variance below this value triggers Fail. Higher = stricter sharpness requirement.",
    )
    blur_review = st.slider(
        "Blur Score - Review below",
        min_value=blur_fail, max_value=600, value=max(int(THRESHOLDS["blur_review"]), blur_fail),
        step=10,
        help="Laplacian variance below this value (but above Fail threshold) triggers Needs Review.",
    )

    overexposure_fail = st.slider(
        "Overexposure % - Fail above",
        min_value=5.0, max_value=60.0, value=float(THRESHOLDS["overexposure_fail"]), step=1.0,
    )
    overexposure_review = st.slider(
        "Overexposure % - Review above",
        min_value=1.0,
        max_value=max(overexposure_fail - 1.0, 1.0),
        value=min(float(THRESHOLDS["overexposure_review"]), overexposure_fail - 1.0),
        step=1.0,
    )

    underexposure_fail = st.slider(
        "Underexposure % - Fail above",
        min_value=5.0, max_value=60.0, value=float(THRESHOLDS["underexposure_fail"]), step=1.0,
    )
    underexposure_review = st.slider(
        "Underexposure % - Review above",
        min_value=1.0,
        max_value=max(underexposure_fail - 1.0, 1.0),
        value=min(float(THRESHOLDS["underexposure_review"]), underexposure_fail - 1.0),
        step=1.0,
    )

    brightness_fail_low = st.slider(
        "Brightness - Fail below",
        min_value=0, max_value=80, value=int(THRESHOLDS["brightness_fail_low"]), step=1,
    )
    brightness_fail_high = st.slider(
        "Brightness - Fail above",
        min_value=150, max_value=255, value=int(THRESHOLDS["brightness_fail_high"]), step=1,
    )

    custom_thresholds = dict(THRESHOLDS)
    custom_thresholds.update({
        "blur_fail": blur_fail,
        "blur_review": blur_review,
        "overexposure_fail": overexposure_fail,
        "overexposure_review": overexposure_review,
        "underexposure_fail": underexposure_fail,
        "underexposure_review": underexposure_review,
        "brightness_fail_low": brightness_fail_low,
        "brightness_fail_high": brightness_fail_high,
    })

    st.markdown("---")
    st.subheader("Metric Definitions")
    with st.expander("View all definitions"):
        st.markdown(
            "**Brightness** - Mean grayscale pixel intensity (0-255). "
            "Reflects overall exposure. Very dark or very bright images indicate poor capture.\n\n"
            "**Contrast** - Standard deviation of pixel intensities. "
            "Low values suggest a flat, washed-out, or featureless image.\n\n"
            "**Blur Score** - Variance of the Laplacian operator applied to the grayscale image. "
            "Higher values indicate sharper images with more fine detail. "
            "Low values indicate blur or out-of-focus acquisition.\n\n"
            "**Overexposure %** - Fraction of pixels at or above intensity 240. "
            "High overexposure saturates bright regions and destroys detail.\n\n"
            "**Underexposure %** - Fraction of pixels at or below intensity 15. "
            "High underexposure crushes shadows and obscures low-signal structure.\n\n"
            "**Edge Density %** - Fraction of pixels classified as edges by Canny detection. "
            "Low values may indicate minimal structural or morphological content.\n\n"
            "**Coverage %** - Estimated foreground fraction via Otsu thresholding. "
            "Useful for gauging stain coverage, cell density, or tissue fill."
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def style_status(val):
    if val == "Pass":
        return "background-color: #d4edda; color: #155724; font-weight: bold"
    elif val == "Needs Review":
        return "background-color: #fff3cd; color: #856404; font-weight: bold"
    elif val == "Fail":
        return "background-color: #f8d7da; color: #721c24; font-weight: bold"
    return ""


def make_status_bar_chart(df):
    counts = [
        (df["QC Status"] == "Pass").sum(),
        (df["QC Status"] == "Needs Review").sum(),
        (df["QC Status"] == "Fail").sum(),
    ]
    labels = ["Pass", "Needs Review", "Fail"]
    colors = ["#28a745", "#ffc107", "#dc3545"]

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(count),
                ha="center", va="bottom", fontsize=10,
            )
    ax.set_ylabel("Count")
    ax.set_title("QC Status Distribution")
    ax.set_ylim(0, max(counts) + 1.5 if max(counts) > 0 else 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def make_metric_histogram(values, title, xlabel, color="#4c72b0"):
    fig, ax = plt.subplots(figsize=(4, 3))
    n_bins = min(20, max(5, len(values)))
    ax.hist(values, bins=n_bins, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def render_image_panel(data):
    col_img, col_metrics = st.columns([2, 3])

    with col_img:
        st.image(data["pil_img"], caption=data["name"], use_container_width=True)

    with col_metrics:
        status = data["status"]
        if status == "Pass":
            st.success(f"QC Status: {status}")
        elif status == "Needs Review":
            st.warning(f"QC Status: {status}")
        else:
            st.error(f"QC Status: {status}")

        if data["flags"]:
            st.markdown("**Flags raised:**")
            for flag in data["flags"]:
                st.markdown(f"- {flag}")
        else:
            st.markdown("No flags raised.")

        st.markdown("---")

        m = data["metrics"]
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Brightness", f"{m['brightness']:.1f}")
        r1c2.metric("Contrast", f"{m['contrast']:.1f}")
        r1c3.metric("Blur Score", f"{m['blur_score']:.1f}")

        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Overexposure %", f"{m['overexposure_pct']:.2f}")
        r2c2.metric("Underexposure %", f"{m['underexposure_pct']:.2f}")
        r2c3.metric("Edge Density %", f"{m['edge_density_pct']:.2f}")

        r3c1, _, _ = st.columns(3)
        r3c1.metric("Coverage %", f"{m['coverage_pct']:.2f}")

        st.markdown("**Pixel Intensity Distribution**")
        hist_df = pd.DataFrame(
            {"Count": data["hist_counts"]},
            index=pd.Index(data["bin_centers"], name="Intensity"),
        )
        st.bar_chart(hist_df, use_container_width=True, height=180)


# ─── Main ─────────────────────────────────────────────────────────────────────

st.title("BioVision QC")
st.caption("Image Quality Control Dashboard")

st.info(
    "This tool computes image quality metrics for lab assay, tissue, and microscopy-style images. "
    "It is intended for research workflow support and portfolio demonstration only. "
    "It does not provide clinical diagnoses or medically validated quality assessments. "
    "Results should not be used for clinical decision-making."
)

# ─── Upload ───────────────────────────────────────────────────────────────────

st.markdown("### Upload Images")
uploaded_files = st.file_uploader(
    "Select images",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    st.markdown("Upload one or more images (JPEG, PNG, TIFF) to begin analysis.")
    st.stop()

# ─── Process ─────────────────────────────────────────────────────────────────

results = []
image_data = []

progress = st.progress(0, text="Analyzing images...")
n_files = len(uploaded_files)

for i, f in enumerate(uploaded_files):
    pil_img = Image.open(f).convert("RGB")
    np_img = np.array(pil_img)

    metrics, bin_centers, hist_counts = analyze_image(np_img)
    status, flags = assign_qc_status(metrics, custom_thresholds)

    results.append({
        "Filename": f.name,
        "Brightness": metrics["brightness"],
        "Contrast": metrics["contrast"],
        "Blur Score": metrics["blur_score"],
        "Overexposure %": metrics["overexposure_pct"],
        "Underexposure %": metrics["underexposure_pct"],
        "Edge Density %": metrics["edge_density_pct"],
        "Coverage %": metrics["coverage_pct"],
        "QC Status": status,
    })

    image_data.append({
        "name": f.name,
        "pil_img": pil_img,
        "metrics": metrics,
        "status": status,
        "flags": flags,
        "bin_centers": bin_centers,
        "hist_counts": hist_counts,
    })

    progress.progress((i + 1) / n_files, text=f"Processed {i + 1}/{n_files}: {f.name}")

progress.empty()
df = pd.DataFrame(results)

# ─── Batch Summary ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Batch Summary")

pass_count = int((df["QC Status"] == "Pass").sum())
review_count = int((df["QC Status"] == "Needs Review").sum())
fail_count = int((df["QC Status"] == "Fail").sum())
avg_blur = float(df["Blur Score"].mean())
avg_brightness = float(df["Brightness"].mean())
avg_contrast = float(df["Contrast"].mean())

sc1, sc2, sc3, sc4, sc5, sc6, sc7 = st.columns(7)
sc1.metric("Total", n_files)
sc2.metric("Pass", pass_count)
sc3.metric("Needs Review", review_count)
sc4.metric("Fail", fail_count)
sc5.metric("Avg Blur", f"{avg_blur:.1f}")
sc6.metric("Avg Brightness", f"{avg_brightness:.1f}")
sc7.metric("Avg Contrast", f"{avg_contrast:.1f}")

chart_c1, chart_c2, chart_c3 = st.columns(3)

with chart_c1:
    fig = make_status_bar_chart(df)
    st.pyplot(fig)
    plt.close(fig)

with chart_c2:
    fig = make_metric_histogram(
        df["Blur Score"].tolist(), "Blur Score Distribution", "Blur Score", color="#4c72b0"
    )
    st.pyplot(fig)
    plt.close(fig)

with chart_c3:
    fig = make_metric_histogram(
        df["Brightness"].tolist(), "Brightness Distribution", "Brightness (0-255)", color="#dd8452"
    )
    st.pyplot(fig)
    plt.close(fig)

# ─── Results Table + Exports ─────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Results Table")

styled = df.style.map(style_status, subset=["QC Status"])
st.dataframe(styled, use_container_width=True, hide_index=True)

export_c1, export_c2 = st.columns(2)

with export_c1:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Results (CSV)",
        data=csv_bytes,
        file_name="biovision_qc_results.csv",
        mime="text/csv",
    )

with export_c2:
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_images": n_files,
        "pass_count": pass_count,
        "review_count": review_count,
        "fail_count": fail_count,
        "averages": {
            "brightness": round(avg_brightness, 2),
            "contrast": round(avg_contrast, 2),
            "blur_score": round(avg_blur, 2),
            "overexposure_pct": round(float(df["Overexposure %"].mean()), 2),
            "underexposure_pct": round(float(df["Underexposure %"].mean()), 2),
            "edge_density_pct": round(float(df["Edge Density %"].mean()), 2),
            "coverage_pct": round(float(df["Coverage %"].mean()), 2),
        },
        "thresholds_used": custom_thresholds,
    }
    json_bytes = json.dumps(summary, indent=2).encode("utf-8")
    st.download_button(
        label="Download Summary Report (JSON)",
        data=json_bytes,
        file_name="biovision_qc_summary.json",
        mime="application/json",
    )

# ─── Per-Image Detail ────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Image Analysis")

if len(image_data) == 1:
    render_image_panel(image_data[0])
else:
    tabs = st.tabs([d["name"] for d in image_data])
    for tab, data in zip(tabs, image_data):
        with tab:
            render_image_panel(data)

# ─── ML Demo ─────────────────────────────────────────────────────────────────

st.markdown("---")

with st.expander("ML Demo - Experimental Classifier", expanded=False):
    st.markdown(
        "This section trains a simple classifier on the image quality metrics above. "
        "It is a demonstration of supervised learning on extracted features, not a validated "
        "quality control system. Human review is required for all decisions."
    )
    st.warning(
        "Limitations: Small datasets overfit easily. Metrics are not assay-specific unless "
        "calibrated to your protocol. Model accuracy on a handful of images is not meaningful. "
        "Do not use predictions from this demo for production QC decisions."
    )

    if n_files < 6:
        st.info(
            f"Upload at least 6 images to enable the classifier demo. "
            f"Currently have {n_files} image(s)."
        )
    else:
        st.markdown("**Assign a label to each image** (defaults to auto-QC status):")
        st.caption(
            "You can override the auto-assigned label. The classifier trains on your labels, "
            "not the auto-QC result."
        )

        n_label_cols = min(4, n_files)
        label_cols = st.columns(n_label_cols)
        labels = {}

        for i, data in enumerate(image_data):
            col = label_cols[i % n_label_cols]
            with col:
                default_idx = ["Pass", "Needs Review", "Fail"].index(data["status"])
                label = st.selectbox(
                    data["name"][:24],
                    options=["Pass", "Needs Review", "Fail"],
                    index=default_idx,
                    key=f"ml_label_{i}",
                )
                labels[data["name"]] = label

        model_type = st.radio(
            "Model type",
            ["Random Forest", "Logistic Regression"],
            horizontal=True,
            key="ml_model_type",
        )

        if st.button("Train Classifier", key="train_classifier_btn"):
            rows = []
            for data in image_data:
                m = data["metrics"]
                rows.append({
                    "Brightness": m["brightness"],
                    "Contrast": m["contrast"],
                    "Blur Score": m["blur_score"],
                    "Overexposure %": m["overexposure_pct"],
                    "Underexposure %": m["underexposure_pct"],
                    "Edge Density %": m["edge_density_pct"],
                    "Coverage %": m["coverage_pct"],
                    "Label": labels[data["name"]],
                })
            labeled_df = pd.DataFrame(rows)

            n_classes = labeled_df["Label"].nunique()
            if n_classes < 2:
                st.error(
                    "All images have the same label. Assign at least 2 different classes to train."
                )
            else:
                with st.spinner("Training..."):
                    result = train_and_evaluate(labeled_df, model_type)
                st.session_state.ml_results = result
                st.session_state.ml_model_type_used = model_type

        if st.session_state.get("ml_results"):
            r = st.session_state.ml_results
            mt = st.session_state.get("ml_model_type_used", "")

            st.markdown(f"**Results ({mt})**")

            ml_c1, ml_c2, ml_c3 = st.columns(3)
            ml_c1.metric("Test Accuracy", f"{r['accuracy']:.1%}")
            ml_c2.metric("Train Samples", r["n_train"])
            ml_c3.metric("Test Samples", r["n_test"])

            st.caption(
                "With small datasets, test accuracy is highly variable. "
                "These numbers reflect in-sample or near-in-sample performance."
            )

            st.markdown("**Classification Report**")
            report_rows = {
                k: v for k, v in r["classification_report"].items()
                if k not in ("accuracy", "macro avg", "weighted avg")
            }
            report_df = pd.DataFrame(report_rows).transpose()[["precision", "recall", "f1-score", "support"]]
            st.dataframe(report_df.round(2), use_container_width=True)

            st.markdown("**Feature Importances**")
            fig = make_feature_importance_chart(r["importances"], r["feature_names"])
            st.pyplot(fig)
            plt.close(fig)
