THRESHOLDS = {
    "brightness_fail_low": 20,
    "brightness_fail_high": 235,
    "brightness_review_low": 40,
    "brightness_review_high": 210,
    "contrast_fail": 10,
    "contrast_review": 20,
    "blur_fail": 50,
    "blur_review": 150,
    "overexposure_fail": 25.0,
    "overexposure_review": 10.0,
    "underexposure_fail": 25.0,
    "underexposure_review": 10.0,
    "edge_density_review": 0.5,
    "coverage_review_low": 5.0,
    "coverage_review_high": 95.0,
}


def assign_qc_status(metrics, thresholds=None):
    t = thresholds if thresholds is not None else THRESHOLDS
    status = "Pass"
    flags = []

    b = metrics["brightness"]
    if b < t["brightness_fail_low"] or b > t["brightness_fail_high"]:
        flags.append("Brightness outside acceptable range")
        status = "Fail"
    elif b < t["brightness_review_low"] or b > t["brightness_review_high"]:
        flags.append("Brightness borderline")
        if status != "Fail":
            status = "Needs Review"

    c = metrics["contrast"]
    if c < t["contrast_fail"]:
        flags.append("Very low contrast")
        status = "Fail"
    elif c < t["contrast_review"]:
        flags.append("Low contrast")
        if status != "Fail":
            status = "Needs Review"

    bl = metrics["blur_score"]
    if bl < t["blur_fail"]:
        flags.append("Image too blurry (low sharpness score)")
        status = "Fail"
    elif bl < t["blur_review"]:
        flags.append("Borderline sharpness")
        if status != "Fail":
            status = "Needs Review"

    oe = metrics["overexposure_pct"]
    if oe > t["overexposure_fail"]:
        flags.append("Excessive overexposure")
        status = "Fail"
    elif oe > t["overexposure_review"]:
        flags.append("Moderate overexposure")
        if status != "Fail":
            status = "Needs Review"

    ue = metrics.get("underexposure_pct", 0.0)
    if ue > t["underexposure_fail"]:
        flags.append("Excessive underexposure")
        status = "Fail"
    elif ue > t["underexposure_review"]:
        flags.append("Moderate underexposure")
        if status != "Fail":
            status = "Needs Review"

    ed = metrics["edge_density_pct"]
    if ed < t["edge_density_review"]:
        flags.append("Very low edge density (minimal structural content detected)")
        if status != "Fail":
            status = "Needs Review"

    cov = metrics["coverage_pct"]
    if cov < t["coverage_review_low"] or cov > t["coverage_review_high"]:
        flags.append("Unusual estimated foreground coverage")
        if status != "Fail":
            status = "Needs Review"

    return status, flags
