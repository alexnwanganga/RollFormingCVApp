"""Streamlit entry point for the roll-forming computer-vision workflow.

This file intentionally owns the end-to-end user workflow: upload image,
detect/crop the tank opening, extract the rim, compute correction guidance,
and render the dashboard. The lower-level image processing helpers live under
``src/``; keep reusable math or CV code there when extending the project.
"""

import cv2                  # Computer vision library for image processing
import json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st      # App framework for building interactive web apps in Python
from PIL import Image       # Python Imaging Library for image manipulation
from io import BytesIO      # For handling in-memory byte streams (used for image upload, specifically camera capture)

from src.detection import load_detector, detect_and_crop_tank
from src.preprocessing import generate_edge_map
from src.rim_analysis import detect_rim_multistart
from src.correction import compute_curvature_correction

# -------------------------------------------------
# STREAMLIT PAGE SETUP
# -------------------------------------------------
st.set_page_config(page_title="Roll Forming CV Analyzer", layout="wide")
st.title("Roll Forming CV Analyzer")

# -------------------------------------------------
# CSS STYLES
# -------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #374151;
        padding: 16px;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }

    div[data-testid="stMetric"] {
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        padding: 14px;
        background-color: var(--secondary-background-color);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
        line-height: 1.05;
    }

    .info-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.3);
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 16px;
    }

    .small-note {
        color: #9ca3af;
        font-size: 0.9rem;
    }

    /* Sidebar slider accent color */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #60a5fa !important;
        border-color: #60a5fa !important;
    }

    .stSlider [data-baseweb="slider"] > div > div > div {
        background-color: #60a5fa !important;
    }

    /* Checkbox accent */
    .stCheckbox input:checked + div {
        background-color: #60a5fa !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write(
    "Upload an image of a rolled shell / tank opening to analyze curvature "
    "and estimate roll-force correction guidance."
)


# -------------------------------------------------
# CACHED PIPELINE FUNCTIONS
# -------------------------------------------------
# Streamlit reruns the full script whenever a widget changes.
# These cached functions prevent expensive operations from rerunning
# unless their inputs actually change.
@st.cache_resource
def get_detector():
    """Load the GroundingDINO detector once per Streamlit process.

    The model is large and may download on first use through Hugging Face.
    Keep this as a resource cache rather than a data cache because the detector
    object is not plain serializable data.
    """
    return load_detector()


@st.cache_data(show_spinner=False)
def cached_detection(image_bytes, threshold):
    """Run object detection on stable bytes so Streamlit can cache the result."""
    image_for_detection = Image.open(BytesIO(image_bytes)).convert("RGB")

    detector = get_detector()

    return detect_and_crop_tank(
        image=image_for_detection,
        detector=detector,
        threshold=threshold
    )


@st.cache_data(show_spinner=False)
def cached_edge_map(
    crop_rgb,
    blur_kernel,
    canny_low,
    canny_high
):
    """Cache Canny edge extraction for a crop and its preprocessing settings."""
    return generate_edge_map(
        crop_rgb=crop_rgb,
        blur_kernel=blur_kernel,
        canny_low=canny_low,
        canny_high=canny_high
    )


@st.cache_data(show_spinner=False)
def cached_rim_detection(
    edges,
    center_x,
    center_y,
    expected_radius,
    search_band,
    max_step_change,
    num_points,
    window_size
):
    """Cache radial rim detection, which is the most parameter-sensitive step."""
    return detect_rim_multistart(
        edges=edges,
        center_x=center_x,
        center_y=center_y,
        expected_radius=expected_radius,
        search_band=search_band,
        max_step_change=max_step_change,
        num_points=num_points,
        window_size=window_size
    )


def fig_to_streamlit(fig):
    """Render a Matplotlib figure and immediately release its memory."""
    st.pyplot(fig)
    plt.close(fig)


def format_signed_term(value, label):
    """Format equation terms without producing awkward '+ -' text."""
    sign = "+" if value >= 0 else "-"
    return f"{sign} {abs(value):.2f}{label}"


def solve_rim_equation(theta_values, radius_values, target_radius_pixels):
    """Fit a compact harmonic model to the radius error around the rim.

    The model decomposes the rim into terms that are meaningful to downstream
    manufacturing discussions:
    - 3theta terms: three-lobed / roll-process variation.
    - 2theta terms: ovality.
    - 1theta terms: off-round, egg-shaped, or remaining center bias.
    - constant term: overall radius offset from the target.

    The fit is linear once the center is fixed, so least squares is sufficient.
    Center refinement happens in ``fit_rim_equation`` below.
    """
    design_matrix = np.column_stack(
        [
            np.cos(3 * theta_values),
            np.sin(3 * theta_values),
            np.cos(2 * theta_values),
            np.sin(2 * theta_values),
            np.cos(theta_values),
            np.sin(theta_values),
            np.ones_like(theta_values),
        ]
    )

    radius_residual = radius_values - target_radius_pixels
    third_cos, third_sin, ovality_cos, ovality_sin, egg_cos, egg_sin, seam_amp = np.linalg.lstsq(
        design_matrix,
        radius_residual,
        rcond=None
    )[0]

    fitted_residual = design_matrix @ np.array(
        [
            third_cos,
            third_sin,
            ovality_cos,
            ovality_sin,
            egg_cos,
            egg_sin,
            seam_amp,
        ]
    )
    fit_error = radius_residual - fitted_residual

    return {
        "third_cos": third_cos,
        "third_sin": third_sin,
        "ovality_cos": ovality_cos,
        "ovality_sin": ovality_sin,
        "egg_cos": egg_cos,
        "egg_sin": egg_sin,
        "seam_amp": seam_amp,
        "third_amp": np.hypot(third_cos, third_sin),
        "ovality_amp": np.hypot(ovality_cos, ovality_sin),
        "egg_amp": np.hypot(egg_cos, egg_sin),
        "third_phase": np.arctan2(third_sin, third_cos) / 3,
        "ovality_phase": np.arctan2(ovality_sin, ovality_cos) / 2,
        "egg_phase": np.arctan2(egg_sin, egg_cos),
        "rmse": np.sqrt(np.mean(fit_error ** 2)),
        "max_error": np.max(np.abs(fit_error)),
    }


def fit_rim_equation(
    x_rim,
    y_rim,
    initial_center_x,
    initial_center_y,
    target_radius_pixels,
    theta_plot
):
    """Refine the center and produce plottable/exportable rim-equation data.

    ``detect_rim_multistart`` starts from the user/manual center. Small center
    errors can masquerade as first-harmonic shape error, so this function tries
    nearby centers and keeps the one with the lowest fit RMSE.
    """
    def fit_for_center(test_center_x, test_center_y):
        dx = x_rim - test_center_x
        dy = y_rim - test_center_y
        theta_values = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        radius_values = np.hypot(dx, dy)

        fit = solve_rim_equation(
            theta_values=theta_values,
            radius_values=radius_values,
            target_radius_pixels=target_radius_pixels
        )
        fit["center_x"] = test_center_x
        fit["center_y"] = test_center_y

        return fit

    # Search only a local neighborhood. A wider search may fit noise or the
    # wrong edge instead of correcting modest manual/detection center error.
    search_radius = max(20.0, target_radius_pixels * 0.04)
    step = max(4.0, search_radius / 4)
    best_fit = fit_for_center(initial_center_x, initial_center_y)

    # Coarse-to-fine grid search keeps this deterministic and dependency-free.
    while step >= 0.5:
        offsets = np.arange(-2, 3) * step

        for offset_y in offsets:
            for offset_x in offsets:
                candidate_center_x = best_fit["center_x"] + offset_x
                candidate_center_y = best_fit["center_y"] + offset_y
                candidate_offset = np.hypot(
                    candidate_center_x - initial_center_x,
                    candidate_center_y - initial_center_y
                )

                if candidate_offset > search_radius:
                    continue

                candidate_fit = fit_for_center(
                    candidate_center_x,
                    candidate_center_y
                )

                if candidate_fit["rmse"] < best_fit["rmse"]:
                    best_fit = candidate_fit

        step /= 2

    best_fit["center_offset"] = np.hypot(
        best_fit["center_x"] - initial_center_x,
        best_fit["center_y"] - initial_center_y
    )

    best_fit["radius_plot"] = (
        target_radius_pixels
        + best_fit["third_cos"] * np.cos(3 * theta_plot)
        + best_fit["third_sin"] * np.sin(3 * theta_plot)
        + best_fit["ovality_cos"] * np.cos(2 * theta_plot)
        + best_fit["ovality_sin"] * np.sin(2 * theta_plot)
        + best_fit["egg_cos"] * np.cos(theta_plot)
        + best_fit["egg_sin"] * np.sin(theta_plot)
        + best_fit["seam_amp"]
    )
    best_fit["x_plot"] = best_fit["center_x"] + best_fit["radius_plot"] * np.cos(theta_plot)
    best_fit["y_plot"] = best_fit["center_y"] + best_fit["radius_plot"] * np.sin(theta_plot)

    return best_fit

# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload rolled shell / tank image (Avoid complex backgrounds for best results)",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.header("Detection Settings")

use_auto_crop = st.sidebar.checkbox("Use Auto-Crop", value=True)

detection_threshold = st.sidebar.slider(
    "Detection Threshold",
    0.05,
    0.90,
    0.25,
    0.05
)

canny_low = st.sidebar.slider("Low Gradient Threshold", 0, 255, 40)
canny_high = st.sidebar.slider("High Gradient Threshold", 0, 255, 120)

blur_kernel = st.sidebar.selectbox(
    "Blur Kernel Size",
    [3, 5, 7, 9, 11, 13],
    index=3
)

st.sidebar.header("Rim Search Settings")

num_points = st.sidebar.slider("Angular Samples", 180, 1440, 720, 180)

search_band = st.sidebar.slider(
    "Search Band [pixels]",
    20,
    300,
    120,
    10
)

max_step_change = st.sidebar.slider(
    "Max Step Change [pixels]",
    5,
    100,
    30,
    5
)

window_size = st.sidebar.slider(
    "Smoothing Window",
    3,
    75,
    21,
    2
)

st.sidebar.header("Curvature Settings")

curvature_tolerance = st.sidebar.number_input(
    "Curvature Tolerance [%]",
    value=3.0,
    step=0.5
)

target_mode = st.sidebar.radio(
    "Target Radius Source",
    [
        "Median detected radius",
        "Expected/manual radius"
    ]
)

real_radius_inches = st.sidebar.number_input(
    "Known Actual Radius [in]",
    value=45.0,
    step=1.0
)

# -------------------------------------------------
# LOAD IMAGE
# -------------------------------------------------
if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()

uploaded_bytes = uploaded_file.getvalue()
image = Image.open(BytesIO(uploaded_bytes)).convert("RGB")

with st.expander("Uploaded Image", expanded=False):
    st.image(image, width=800)

# -------------------------------------------------
# AUTO CROP USING src/detection.py
# -------------------------------------------------

if use_auto_crop:

    detection_output = cached_detection(
        uploaded_bytes,
        detection_threshold
    )

    results = detection_output["results"]
    annotated_image = detection_output["annotated_image"]
    crop = detection_output["crop"]
    best_detection = detection_output["best_detection"]

    if best_detection is None:
        # Analysis can still proceed on the whole image; the manual geometry
        # controls become more important when auto-crop fails.
        st.warning("No tank region detected. Using full image.")
        crop = image
    else:
        with st.expander("Detected Tank Region", expanded=False):
            st.image(annotated_image, width=800)

else:
    crop = image
    results = []
    best_detection = None

crop_rgb = np.array(crop)

with st.expander("Analysis Crop", expanded=False):
    st.image(crop_rgb, width=800)

# -------------------------------------------------
# EDGE DETECTION using src/preprocessing.py
# -------------------------------------------------

edge_output = cached_edge_map(
    crop_rgb,
    blur_kernel,
    canny_low,
    canny_high
)

crop_gray = edge_output["gray"]
crop_blur = edge_output["blurred"]
edges = edge_output["edges"]

with st.expander("Edge Detection", expanded=False):
    st.image(edges, clamp=True, width=800)

# -------------------------------------------------
# APPROXIMATE GEOMETRY using src/rim_analysis.py
# -------------------------------------------------

height, width = edges.shape

default_center_x = width // 2
default_center_y = height // 2
default_radius = int(0.5 * min(width, height))

st.sidebar.header("Manual Geometry Override")

center_x = st.sidebar.number_input(
    "Center X [pixels]",
    value=int(default_center_x),
    step=10
)

center_y = st.sidebar.number_input(
    "Center Y [pixels]",
    value=int(default_center_y),
    step=10
)

expected_radius = st.sidebar.number_input(
    "Expected Radius [pixels]",
    value=int(default_radius),
    step=10
)

center_x = int(center_x)
center_y = int(center_y)
expected_radius = int(expected_radius)

# -------------------------------------------------
# MULTI-START RADIAL RIM DETECTION using src/rim_analysis.py
# -------------------------------------------------

rim_output = cached_rim_detection(
    edges,
    center_x,
    center_y,
    expected_radius,
    search_band,
    max_step_change,
    num_points,
    window_size
)

theta_uniform = rim_output["theta_uniform"]
radius_uniform_pixels = rim_output["radius_uniform_pixels"]

x_rim = rim_output["x_rim"]
y_rim = rim_output["y_rim"]

circle_x = rim_output["circle_x"]
circle_y = rim_output["circle_y"]

all_radius_results = rim_output["all_radius_results"]

# -------------------------------------------------
# CURVATURE ERROR + FORCE CORRECTION using src/correction.py
# -------------------------------------------------

correction_output = compute_curvature_correction(
    radius_uniform_pixels=radius_uniform_pixels,
    theta_uniform=theta_uniform,
    expected_radius=expected_radius,
    real_radius_inches=real_radius_inches,
    curvature_tolerance=curvature_tolerance,
    target_mode=target_mode
)

pixels_per_inch = correction_output["pixels_per_inch"]

target_radius_pixels = correction_output["target_radius_pixels"]
target_radius_inches = correction_output["target_radius_inches"]

# The lower-level correction helper uses the mathematical curvature sign.
# The dashboard flips signs so positive values line up with operator-facing
# "increase force / too flat" guidance. Preserve this convention when changing
# labels, legends, or exported correction maps.
curvature_error_percent = -correction_output["curvature_error_percent"]
force_correction_percent = -correction_output["force_correction_percent"]

too_flat = correction_output["too_flat"]
too_tight = correction_output["too_tight"]
acceptable = correction_output["acceptable"]

rim_fit = fit_rim_equation(
    x_rim=x_rim,
    y_rim=y_rim,
    initial_center_x=center_x,
    initial_center_y=center_y,
    target_radius_pixels=target_radius_pixels,
    theta_plot=theta_uniform
)

third_cos_pixels = rim_fit["third_cos"]
third_sin_pixels = rim_fit["third_sin"]
ovality_cos_pixels = rim_fit["ovality_cos"]
ovality_sin_pixels = rim_fit["ovality_sin"]
egg_cos_pixels = rim_fit["egg_cos"]
egg_sin_pixels = rim_fit["egg_sin"]
seam_amp_pixels = rim_fit["seam_amp"]
third_amp_pixels = rim_fit["third_amp"]
ovality_amp_pixels = rim_fit["ovality_amp"]
egg_amp_pixels = rim_fit["egg_amp"]
third_phase = rim_fit["third_phase"]
ovality_phase = rim_fit["ovality_phase"]
egg_phase = rim_fit["egg_phase"]

rim_equation_pixels = (
    f"R(θ) = {target_radius_pixels:.2f} "
    f"{format_signed_term(third_cos_pixels, 'cos(3θ)')} "
    f"{format_signed_term(third_sin_pixels, 'sin(3θ)')} "
    f"{format_signed_term(ovality_cos_pixels, 'cos(2θ)')} "
    f"{format_signed_term(ovality_sin_pixels, 'sin(2θ)')} "
    f"{format_signed_term(egg_cos_pixels, 'cos(θ)')} "
    f"{format_signed_term(egg_sin_pixels, 'sin(θ)')} "
    f"{format_signed_term(seam_amp_pixels, '')}"
)

derived_x_rim = rim_fit["x_plot"]
derived_y_rim = rim_fit["y_plot"]
equation_center_x = rim_fit["center_x"]
equation_center_y = rim_fit["center_y"]

rim_equation_export = {
    "equation_name": "Derived Rim Equation",
    "equation_form": (
        "R(theta) = Rt + A3c*cos(3*theta) + A3s*sin(3*theta) "
        "+ A0c*cos(2*theta) + A0s*sin(2*theta) "
        "+ Aec*cos(theta) + Aes*sin(theta) + As"
    ),
    # Exported coefficients are converted to inches so they can be consumed
    # outside the pixel-based UI by process engineers or downstream tooling.
    "correction_form": "correction(theta) = Rt - R(theta)",
    "units": {
        "radius": "inches",
        "theta": "radians"
    },
    "coefficients": {
        "Rt": float(target_radius_inches),
        "A3c": float(third_cos_pixels / pixels_per_inch),
        "A3s": float(third_sin_pixels / pixels_per_inch),
        "A0c": float(ovality_cos_pixels / pixels_per_inch),
        "A0s": float(ovality_sin_pixels / pixels_per_inch),
        "Aec": float(egg_cos_pixels / pixels_per_inch),
        "Aes": float(egg_sin_pixels / pixels_per_inch),
        "As": float(seam_amp_pixels / pixels_per_inch),
    },
    "amplitude_phase": {
        "A3": float(third_amp_pixels / pixels_per_inch),
        "phi3": float(third_phase),
        "A0": float(ovality_amp_pixels / pixels_per_inch),
        "phi0": float(ovality_phase),
        "Ae": float(egg_amp_pixels / pixels_per_inch),
        "phie": float(egg_phase),
    },
    "fit_quality": {
        "rmse": float(rim_fit["rmse"] / pixels_per_inch),
        "max_error": float(rim_fit["max_error"] / pixels_per_inch),
    },
}

rim_equation_json = json.dumps(rim_equation_export, indent=2)
rim_equation_csv = "\n".join(
    [
        "name,value,units",
        f"Rt,{target_radius_inches:.10g},inches",
        f"A3c,{third_cos_pixels / pixels_per_inch:.10g},inches",
        f"A3s,{third_sin_pixels / pixels_per_inch:.10g},inches",
        f"A0c,{ovality_cos_pixels / pixels_per_inch:.10g},inches",
        f"A0s,{ovality_sin_pixels / pixels_per_inch:.10g},inches",
        f"Aec,{egg_cos_pixels / pixels_per_inch:.10g},inches",
        f"Aes,{egg_sin_pixels / pixels_per_inch:.10g},inches",
        f"As,{seam_amp_pixels / pixels_per_inch:.10g},inches",
        f"rmse,{rim_fit['rmse'] / pixels_per_inch:.10g},inches",
        f"max_error,{rim_fit['max_error'] / pixels_per_inch:.10g},inches",
    ]
)

# -------------------------------------------------
# SUMMARY
# -------------------------------------------------

st.subheader("Curvature / Force Correction Summary")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Target Radius", f"{target_radius_inches:.2f} in")
col2.metric("Target Radius", f"{target_radius_pixels:.1f} px")
col3.metric("Within Tolerance", f"{correction_output['within_tolerance_percent']:.1f}%")
col4.metric("Too Tight", f"{correction_output['too_flat_percent']:.1f}%")
col5.metric("Too Flat", f"{correction_output['too_tight_percent']:.1f}%")
col6.metric("Pixels/Inch", f"{pixels_per_inch:.2f}")

force_col1, force_col2 = st.columns(2)

with force_col1:
    st.markdown(
        f"""
        <div class="info-card">
        <strong>Largest Force Decrease</strong><br>
        <span class="small-note">Too tight / over-bent region</span><br><br>
        <span style="font-size: 1.4rem; color: #f87171;">
        {-correction_output['max_force_increase']:.2f}% at θ = {correction_output['max_force_increase_angle']:.3f} rad
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

with force_col2:
    st.markdown(
        f"""
        <div class="info-card">
        <strong>Largest Force Increase</strong><br>
        <span class="small-note">Too flat / needs more bending</span><br><br>
        <span style="font-size: 1.4rem; color: #60a5fa;">
        {-correction_output['max_force_decrease']:.2f}% at θ = {correction_output['max_force_decrease_angle']:.3f} rad
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# DASHBOARD PLOTS
# -------------------------------------------------

main_left, main_right = st.columns([1.35, 1])

with main_left:
    st.markdown("### Correction Zones")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.imshow(crop_rgb)

    ax.scatter(
        x_rim[acceptable],
        y_rim[acceptable],
        s=5,
        color="lime",
        #label=f"Within ±{curvature_tolerance:.1f}%"
    )

    ax.scatter(
        x_rim[too_tight],
        y_rim[too_tight],
        s=7,
        color="blue",
        #label="Too Flat"
    )

    ax.scatter(
        x_rim[too_flat],
        y_rim[too_flat],
        s=7,
        color="red",
        #label="Too Tight"
    )

    ax.axis("equal")
    ax.legend()
    ax.set_title("Correction Zones")

    fig_to_streamlit(fig)

with main_right:
    st.markdown(
        """
        <div class="info-card">
        <h4>Legend</h4>
        <p>🟢 Within tolerance / detected rim</p>
        <p>🔴 Too tight: decrease force</p>
        <p>🔵 Too flat: increase force</p>
        <p><span style="color:red;">- - -</span> Expected/manual circle</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        st.download_button(
            label="Download Equation JSON",
            data=rim_equation_json,
            file_name="derived_rim_equation.json",
            mime="application/json"
        )

    with export_col2:
        st.download_button(
            label="Download Coefficients CSV",
            data=rim_equation_csv,
            file_name="derived_rim_coefficients.csv",
            mime="text/csv"
        )

    st.markdown(
        f"""
        <div class="info-card">
        <h4>Derived Rim Equation</h4>
        <p><strong>R(θ) = Rₜ + A₃c cos(3θ) + A₃s sin(3θ) + A₀c cos(2θ) + A₀s sin(2θ) + Aₑc cos(θ) + Aₑs sin(θ) + Aₛ</strong></p>
        <p class="small-note">
        Fitted from detected rim points:
        Rₜ = {target_radius_pixels:.2f} px,
        A₃ = {third_amp_pixels:.2f} px at φ₃ = {third_phase:.3f} rad,
        A₀ = {ovality_amp_pixels:.2f} px at φ₀ = {ovality_phase:.3f} rad,
        Aₑ = {egg_amp_pixels:.2f} px at φₑ = {egg_phase:.3f} rad,
        Aₛ = {seam_amp_pixels:.2f} px.
        </p>
        <p class="small-note">
        Fit error: RMSE = {rim_fit["rmse"]:.2f} px,
        max = {rim_fit["max_error"]:.2f} px,
        center adjustment = {rim_fit["center_offset"]:.2f} px.
        </p>
        <p class="small-note">{rim_equation_pixels}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="info-card">
        <h4>Notes</h4>
        <p class="small-note">
        Rim detected using multi-start radial edge search with circular smoothing.
        Curvature tolerance is ±{curvature_tolerance:.1f}%.
        Force correction is estimated from local curvature error.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

plot1, plot2, plot3 = st.columns(3)

with plot1:
    st.markdown("### Curvature Error")

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(
        theta_uniform,
        curvature_error_percent,
        color="lime",
        label="Curvature Error"
    )

    ax.axhline(
        curvature_tolerance,
        linestyle="--",
        color="red",
        label=f"+{curvature_tolerance:.1f}%"
    )

    ax.axhline(
        -curvature_tolerance,
        linestyle="--",
        color="red",
        label=f"-{curvature_tolerance:.1f}%"
    )

    ax.axhline(
        0,
        linestyle="--",
        color="gray",
        label="0%"
    )

    ax.set_xlabel("Angle θ [rad]")
    ax.set_ylabel("Error [%]")
    ax.set_title("Curvature Error Around Shell")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig_to_streamlit(fig)

with plot2:
    st.markdown("### Force Correction")

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(
        theta_uniform,
        force_correction_percent,
        label="Force Correction"
    )

    ax.axhline(
        0,
        linestyle="--",
        color="gray",
        label="0%"
    )

    ax.set_xlabel("Angle θ [rad]")
    ax.set_ylabel("Correction [%]")
    ax.set_title("Estimated Force Correction")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig_to_streamlit(fig)

with plot3:
    st.markdown("### Derived Rim Equation")

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.imshow(crop_rgb)
    ax.scatter(
        x_rim,
        y_rim,
        s=4,
        color="red",
        label="Detected Rim"
    )
    ax.plot(
        derived_x_rim,
        derived_y_rim,
        color="lime",
        linewidth=2,
        label="Derived Rim Equation"
    )
    ax.scatter(
        equation_center_x,
        equation_center_y,
        color="lime",
        s=35,
        label="Equation Center"
    )

    ax.axis("equal")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=8
    )
    ax.set_title("Derived Rim Equation")
    fig.tight_layout()

    fig_to_streamlit(fig)

# -------------------------------------------------
# FOOTNOTE
# -------------------------------------------------

st.caption(
    "Force correction is estimated from local curvature error. "
    "This is a guidance tool, not a calibrated roll-forming force prediction."
)
