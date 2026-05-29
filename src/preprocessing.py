import cv2
import numpy as np


def generate_edge_map(
    crop_rgb,
    blur_kernel=9,
    canny_low=40,
    canny_high=120
):
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    crop_gray = cv2.cvtColor(
        crop_rgb,
        cv2.COLOR_RGB2GRAY
    )

    crop_blur = cv2.GaussianBlur(
        crop_gray,
        (blur_kernel, blur_kernel),
        0
    )

    edges = cv2.Canny(
        crop_blur,
        threshold1=canny_low,
        threshold2=canny_high
    )

    return {
        "gray": crop_gray,
        "blurred": crop_blur,
        "edges": edges
    }