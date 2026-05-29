import cv2                  # Computer vision library for image processing
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
    return load_detector()


@st.cache_data(show_spinner=False)
def cached_detection(image_bytes, threshold):
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
    st.pyplot(fig)
    plt.close(fig)

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

curvature_error_percent = -correction_output["curvature_error_percent"]
force_correction_percent = -correction_output["force_correction_percent"]

too_flat = correction_output["too_flat"]
too_tight = correction_output["too_tight"]
acceptable = correction_output["acceptable"]

# -------------------------------------------------
# SUMMARY
# -------------------------------------------------

st.subheader("Curvature / Force Correction Summary")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Target Radius", f"{target_radius_inches:.2f} in")
col2.metric("Target Radius", f"{target_radius_pixels:.1f} px")
col3.metric("Within Tolerance", f"{correction_output['within_tolerance_percent']:.1f}%")
col4.metric("Too Flat", f"{correction_output['too_flat_percent']:.1f}%")
col5.metric("Too Tight", f"{correction_output['too_tight_percent']:.1f}%")
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
    st.markdown("### Detected Rim")

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.imshow(crop_rgb)
    ax.plot(circle_x, circle_y, "r--", label="Expected Circle")
    ax.scatter(x_rim, y_rim, s=4, color="lime", label="Detected Rim")
    ax.scatter(center_x, center_y, color="lime", s=35, label="Center")

    ax.axis("equal")
    ax.legend(fontsize=8)
    ax.set_title("Detected Rim")

    fig_to_streamlit(fig)

# -------------------------------------------------
# FOOTNOTE
# -------------------------------------------------

st.caption(
    "Force correction is estimated from local curvature error. "
    "This is a guidance tool, not a calibrated roll-forming force prediction."
)
