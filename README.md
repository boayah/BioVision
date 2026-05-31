# BioVision QC

An image quality control dashboard for lab assay, tissue, and microscopy-style images.
Built with Python, Streamlit, OpenCV, and scikit-image.

> This is a portfolio and research workflow demonstration project.
> It does not provide clinical diagnoses or medically validated quality assessments.

---

## Why This Exists

Manual image QC in lab workflows is repetitive and hard to standardize across operators or imaging sessions. Small differences in acquisition settings, staining protocol, or sample prep produce images that are technically unusable but still get annotated or analyzed downstream. This project applies classical computer vision metrics to automate that first-pass triage and flag images that need review before they enter a pipeline.

The same problem shows up in high-content screening, histopathology slide digitization, and any automated assay where image quality is a precondition for downstream analysis. BioVision QC demonstrates a practical approach to it using open tools and configurable thresholds.

---

## Features

- Upload one or more images (JPEG, PNG, TIFF)
- Compute seven image quality metrics per image
- Assign QC status: Pass, Needs Review, or Fail
- Batch-level summary: counts, averages, distribution charts
- Per-image detail view with pixel intensity histogram
- Adjustable thresholds in the sidebar
- Export full results as CSV
- Export batch summary as JSON
- Optional experimental ML classifier trained on image metrics

---

## Metrics

| Metric | Method | What it reflects |
|---|---|---|
| Brightness | Mean grayscale intensity (0-255) | Overall exposure level |
| Contrast | Std dev of pixel intensities | Tonal range / image flatness |
| Blur Score | Variance of Laplacian | Sharpness / focus quality |
| Overexposure % | Fraction of pixels >= 240 | Saturated bright regions |
| Underexposure % | Fraction of pixels <= 15 | Crushed shadow regions |
| Edge Density % | Canny edge pixel fraction | Structural content present |
| Coverage % | Otsu-threshold foreground fraction | Stain or cell fill estimate |

---

## QC Rules

| Status | Condition |
|---|---|
| Fail | Blur too low, contrast too low, brightness out of range, or overexposure/underexposure exceeds fail threshold |
| Needs Review | Any metric in borderline range, low edge density, or unusual coverage |
| Pass | All metrics within acceptable ranges |

Default thresholds are general-purpose starting points. For meaningful results, calibrate thresholds against representative good and bad images from your specific assay and imaging setup.

---

## Architecture

```
BioVision/
├── app.py                  # Streamlit entrypoint: UI, layout, state management
├── requirements.txt
├── README.md
├── sample_data/            # Place test images here
└── utils/
    ├── image_metrics.py    # Metric computation (OpenCV, scikit-image, numpy)
    ├── qc_rules.py         # Threshold-based QC logic and status assignment
    └── ml_demo.py          # Experimental classifier (scikit-learn)
```

Data flow:

```
Uploaded image (PIL)
  -> numpy array (RGB)
  -> grayscale conversion (OpenCV)
  -> compute_brightness, compute_contrast, compute_blur_score,
     compute_overexposure, compute_underexposure,
     compute_edge_density, compute_coverage
  -> assign_qc_status (threshold comparison)
  -> Streamlit display + DataFrame
  -> CSV / JSON export
```

---

## Example Workflow

1. Collect images from your microscope or plate reader
2. Upload them to BioVision QC
3. Review the batch summary to get a sense of overall acquisition quality
4. Check the results table for any Fail or Needs Review images
5. Open the per-image tab for flagged images and look at the intensity distribution
6. Adjust thresholds in the sidebar to match your protocol's acceptable range
7. Export the CSV and attach it to your experiment log
8. Optionally: use the ML demo to see which metrics are most predictive of human QC labels

---

## ML Demo

The ML demo section is an experimental feature. It trains a scikit-learn classifier (Logistic Regression or Random Forest) on the computed image metrics, using labels you assign manually. It requires at least 6 labeled images with at least 2 distinct classes.

It is intended to show how feature-based ML can augment or accelerate QC workflows, not to replace human review. Accuracy on small datasets is not statistically meaningful.

---

## Limitations

- Blur score is sensitive to image content. A flat, featureless field will score low even if it is in perfect focus.
- Coverage estimation via Otsu thresholding assumes a bimodal intensity distribution. It degrades on images that are uniformly bright or dark.
- Default thresholds were not derived from any specific assay. They are reasonable starting points, not validated cutoffs.
- The ML demo overfits severely with fewer than ~30 labeled examples.
- TIFF files with unusual bit depths or multi-channel formats are converted to 8-bit RGB on load, which may affect metrics.

---

## Future Work

- Per-channel analysis for fluorescence multi-channel images
- Batch comparison across imaging sessions or plates
- Integration with OMERO, QuPath, or CellProfiler output formats
- Threshold auto-calibration using a labeled reference set
- Report generation (PDF)
- REST API wrapper for pipeline integration

---

## Setup

```bash
pip install -r requirements.txt --break-system-packages
streamlit run app.py
```

Python 3.10+ required.

---

## Screenshots

*(Add screenshots of the batch summary and per-image panel here)*

---

## Resume Bullets

- Built an image quality control dashboard in Python (Streamlit, OpenCV, scikit-image) that computes seven signal quality metrics on lab and microscopy images and assigns tiered QC status with configurable thresholds
- Implemented batch-level reporting with matplotlib distribution charts and exports (CSV, JSON) to support lab documentation and reproducible analysis workflows
- Integrated an experimental scikit-learn classifier (Logistic Regression, Random Forest) to demonstrate how extracted image features can augment manual QC labeling at scale

---

## Interview Talking Points

**Why image QC matters in a lab context**
Bad images that pass visual inspection by a tired operator are a real problem in high-throughput workflows. Automating the first pass with quantifiable metrics reduces inter-operator variability and creates an audit trail. This is the same reason CellProfiler includes QC modules and high-content screening vendors build flagging logic into their acquisition software.

**Why these specific metrics**
Blur (Laplacian variance), brightness, and contrast are the minimum viable metrics for exposure and focus quality. Overexposure and underexposure catch saturation artifacts that are invisible in a casual visual review. Edge density and coverage are domain-informed: in cell biology and histology, images with very low edge density often indicate empty fields or failed staining, and coverage correlates with confluency or tissue fill.

**Threshold calibration**
The defaults in this project are not validated for any assay. In a production context, you would derive thresholds from a labeled reference set of acceptable and unacceptable images from your specific protocol. That is a supervised calibration problem, which is what the ML demo is hinting at.

**The ML section**
The classifier demo makes a real point: once you have enough human-labeled examples, you can learn a decision boundary over these features that is specific to your assay. The thresholds in qc_rules.py are equivalent to a hand-tuned axis-aligned classifier. A learned model can capture interactions between metrics (e.g., an image can be slightly blurry and slightly dark but still acceptable, as long as it is not both). The feature importance chart makes the model's reasoning visible, which is useful for understanding which aspects of image quality are actually predictive in a given context.

**Tech choices**
OpenCV for classical image processing (Laplacian, Canny), scikit-image for Otsu thresholding (more robust implementation), pandas for the results table, matplotlib for batch charts (no seaborn dependency), Streamlit for the interface because it is the fastest path from Python code to a shareable dashboard. No web framework, no frontend build step.
