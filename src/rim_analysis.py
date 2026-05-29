import numpy as np


def detect_rim_multistart(
    edges,
    center_x,
    center_y,
    expected_radius,
    search_band=120,
    max_step_change=30,
    num_points=720,
    window_size=21,
):
    base_theta = np.linspace(
        0,
        2 * np.pi,
        num_points,
        endpoint=False
    )

    start_angles = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
        np.pi,
        5 * np.pi / 4,
        3 * np.pi / 2,
        7 * np.pi / 4,
    ]

    inner_radius = expected_radius - search_band
    outer_radius = expected_radius + search_band

    all_radius_results = []

    for start_angle in start_angles:
        theta_scan = np.linspace(
            start_angle,
            start_angle + 2 * np.pi,
            num_points,
            endpoint=False
        )

        theta_wrapped_all = np.mod(theta_scan, 2 * np.pi)

        detected_radii = []
        previous_radius = expected_radius

        for theta in theta_scan:
            theta_wrapped = np.mod(theta, 2 * np.pi)

            radii = np.arange(inner_radius, outer_radius)

            xs = (
                center_x + radii * np.cos(theta_wrapped)
            ).astype(int)

            ys = (
                center_y + radii * np.sin(theta_wrapped)
            ).astype(int)

            valid = (
                (xs >= 0)
                & (xs < edges.shape[1])
                & (ys >= 0)
                & (ys < edges.shape[0])
            )

            xs = xs[valid]
            ys = ys[valid]
            radii_valid = radii[valid]

            edge_values = edges[ys, xs]
            edge_indices = np.where(edge_values > 0)[0]

            if len(edge_indices) > 0:
                candidate_radii = radii_valid[edge_indices]

                selected_radius = candidate_radii[
                    np.argmin(
                        np.abs(candidate_radii - previous_radius)
                    )
                ]

                if abs(selected_radius - previous_radius) <= max_step_change:
                    detected_radii.append(selected_radius)
                    previous_radius = selected_radius
                else:
                    detected_radii.append(previous_radius)
            else:
                detected_radii.append(previous_radius)

        detected_radii = np.array(detected_radii)

        sort_index = np.argsort(theta_wrapped_all)

        detected_sorted = detected_radii[sort_index]

        all_radius_results.append(detected_sorted)

    all_radius_results = np.array(all_radius_results)

    radius_uniform_pixels = np.median(
        all_radius_results,
        axis=0
    )

    theta_uniform = base_theta

    radius_uniform_pixels = circular_smooth(
        radius_uniform_pixels,
        window_size=window_size
    )

    x_rim, y_rim = generate_rim_coordinates(
        center_x=center_x,
        center_y=center_y,
        radius_uniform_pixels=radius_uniform_pixels,
        theta_uniform=theta_uniform
    )

    circle_x, circle_y = generate_circle_coordinates(
        center_x=center_x,
        center_y=center_y,
        radius=expected_radius,
        theta_uniform=theta_uniform
    )

    return {
        "theta_uniform": theta_uniform,
        "radius_uniform_pixels": radius_uniform_pixels,
        "x_rim": x_rim,
        "y_rim": y_rim,
        "circle_x": circle_x,
        "circle_y": circle_y,
        "all_radius_results": all_radius_results,
    }


def circular_smooth(radius_values, window_size=21):
    if window_size % 2 == 0:
        window_size += 1

    kernel = np.ones(window_size) / window_size
    pad = window_size // 2

    radius_padded = np.pad(
        radius_values,
        pad_width=pad,
        mode="wrap"
    )

    smoothed = np.convolve(
        radius_padded,
        kernel,
        mode="valid"
    )

    return smoothed


def generate_rim_coordinates(
    center_x,
    center_y,
    radius_uniform_pixels,
    theta_uniform
):
    x_rim = (
        center_x
        + radius_uniform_pixels * np.cos(theta_uniform)
    )

    y_rim = (
        center_y
        + radius_uniform_pixels * np.sin(theta_uniform)
    )

    return x_rim, y_rim


def generate_circle_coordinates(
    center_x,
    center_y,
    radius,
    theta_uniform
):
    circle_x = center_x + radius * np.cos(theta_uniform)
    circle_y = center_y + radius * np.sin(theta_uniform)

    return circle_x, circle_y