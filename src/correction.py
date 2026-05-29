import numpy as np


def compute_curvature_correction(
    radius_uniform_pixels,
    theta_uniform,
    expected_radius,
    real_radius_inches,
    curvature_tolerance=3.0,
    target_mode="Median detected radius",
):
    pixels_per_inch = expected_radius / real_radius_inches

    if target_mode == "Median detected radius":
        target_radius_pixels = np.median(radius_uniform_pixels)
    elif target_mode == "Expected/manual radius":
        target_radius_pixels = expected_radius
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    target_radius_inches = target_radius_pixels / pixels_per_inch
    actual_radius_inches = radius_uniform_pixels / pixels_per_inch

    target_curvature = 1 / target_radius_pixels
    actual_curvature = 1 / radius_uniform_pixels

    curvature_error_percent = (
        (actual_curvature - target_curvature)
        / target_curvature
    ) * 100

    force_correction_percent = (
        (target_curvature - actual_curvature)
        / actual_curvature
    ) * 100

    # Lower curvature means larger radius / flatter section.
    too_flat = actual_curvature < target_curvature * (1 - curvature_tolerance / 100)

    # Higher curvature means smaller radius / tighter section.
    too_tight = actual_curvature > target_curvature * (1 + curvature_tolerance / 100)

    acceptable = ~(too_flat | too_tight)

    max_increase_idx = np.argmax(force_correction_percent)
    max_decrease_idx = np.argmin(force_correction_percent)

    return {
        "pixels_per_inch": pixels_per_inch,
        "target_radius_pixels": target_radius_pixels,
        "target_radius_inches": target_radius_inches,
        "actual_radius_inches": actual_radius_inches,
        "target_curvature": target_curvature,
        "actual_curvature": actual_curvature,
        "curvature_error_percent": curvature_error_percent,
        "force_correction_percent": force_correction_percent,
        "too_flat": too_flat,
        "too_tight": too_tight,
        "acceptable": acceptable,
        "within_tolerance_percent": 100 * np.mean(acceptable),
        "too_flat_percent": 100 * np.mean(too_flat),
        "too_tight_percent": 100 * np.mean(too_tight),
        "max_increase_idx": max_increase_idx,
        "max_decrease_idx": max_decrease_idx,
        "max_force_increase": force_correction_percent[max_increase_idx],
        "max_force_decrease": force_correction_percent[max_decrease_idx],
        "max_force_increase_angle": theta_uniform[max_increase_idx],
        "max_force_decrease_angle": theta_uniform[max_decrease_idx],
    }