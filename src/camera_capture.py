import time
import cv2
import streamlit as st
from PIL import Image


@st.cache_resource
def get_camera(camera_index=0):
    return cv2.VideoCapture(camera_index)


def draw_alignment_circle(
    frame,
    guide_radius_ratio=0.38,
    guide_color=(96, 165, 250),
    guide_thickness=3
):
    height, width = frame.shape[:2]

    center_x = width // 2
    center_y = height // 2

    radius = int(min(width, height) * guide_radius_ratio)

    cv2.circle(
        frame,
        (center_x, center_y),
        radius,
        guide_color,
        guide_thickness
    )

    cv2.circle(
        frame,
        (center_x, center_y),
        5,
        guide_color,
        -1
    )

    return frame


def render_camera_capture(
    guide_radius_ratio=0.38,
    guide_color=(96, 165, 250),
    guide_thickness=3
):
    st.subheader("Camera Capture")

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    if "captured_camera_image" not in st.session_state:
        st.session_state.captured_camera_image = None

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Start Camera"):
            st.session_state.camera_running = True

    with col2:
        capture_clicked = st.button("Capture Frame")

    with col3:
        if st.button("Stop Camera"):
            st.session_state.camera_running = False

    camera_placeholder = st.empty()

    if st.session_state.camera_running:
        cap = get_camera(0)

        ret, frame = cap.read()

        if not ret:
            st.warning("Could not read from camera.")
            return st.session_state.captured_camera_image

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        raw_frame = frame.copy()

        preview_frame = draw_alignment_circle(
            frame=frame.copy(),
            guide_radius_ratio=guide_radius_ratio,
            guide_color=guide_color,
            guide_thickness=guide_thickness
        )

        camera_placeholder.image(
            preview_frame,
            caption="Live camera feed with alignment guide",
            use_container_width=True
        )

        if capture_clicked:
            st.session_state.captured_camera_image = Image.fromarray(raw_frame)
            st.session_state.camera_running = False
            st.success("Frame captured.")

        time.sleep(0.05)
        st.rerun()

    if st.session_state.captured_camera_image is not None:
        st.image(
            st.session_state.captured_camera_image,
            caption="Captured frame for analysis",
            width=800
        )

    return st.session_state.captured_camera_image