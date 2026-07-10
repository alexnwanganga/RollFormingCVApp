"""Object-detection utilities for locating the tank/rim region.

The app uses a zero-shot detector so the project can recognize likely tank
openings without training a custom model. If accuracy becomes inconsistent in
production, this is the module to replace with a fine-tuned detector or a more
controlled segmentation pipeline.
"""

import cv2
import numpy as np
from transformers import pipeline


DEFAULT_TANK_LABELS = [
    # GroundingDINO is sensitive to wording. Keep several near-synonyms so one
    # weak phrase does not prevent a usable crop on shop-floor photos.
    "large circular metal tank opening",
    "steel cylinder opening",
    "round metal shell",
    "metal tank rim",
]


def load_detector():
    """Create the Hugging Face zero-shot detector used by Streamlit caching."""
    detector = pipeline(
        model="IDEA-Research/grounding-dino-base",
        task="zero-shot-object-detection"
    )
    return detector


def detect_tank_region(
    image,
    detector,
    candidate_labels=None,
    threshold=0.25
):
    """Return raw detector boxes for candidate tank-opening labels.

    Expected ``results`` item shape from transformers:
    ``{"score": float, "label": str, "box": {"xmin", "ymin", "xmax", "ymax"}}``.
    Downstream helpers rely on that structure, so update them together if the
    detector/model API changes.
    """
    if candidate_labels is None:
        candidate_labels = DEFAULT_TANK_LABELS

    results = detector(
        image,
        candidate_labels=candidate_labels,
        threshold=threshold
    )

    return results


def draw_detection_boxes(image, results):
    """Draw detector output for the optional UI expander."""
    image_np = np.array(image).copy()

    for result in results:
        box = result["box"]
        label = result["label"]
        score = result["score"]

        x_min = int(box["xmin"])
        y_min = int(box["ymin"])
        x_max = int(box["xmax"])
        y_max = int(box["ymax"])

        cv2.rectangle(
            image_np,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            4
        )

        cv2.putText(
            image_np,
            f"{label}: {score:.2f}",
            (x_min, max(y_min - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return image_np


def crop_best_detection(image, results):
    """Crop to the highest-confidence detection.

    If no detection is available, return the original image and ``None`` so the
    caller can continue with manual geometry instead of failing the whole run.
    """
    if len(results) == 0:
        return image, None

    best = max(results, key=lambda r: r["score"])
    box = best["box"]

    x_min = int(box["xmin"])
    y_min = int(box["ymin"])
    x_max = int(box["xmax"])
    y_max = int(box["ymax"])

    crop = image.crop(
        (x_min, y_min, x_max, y_max)
    )

    return crop, best


def detect_and_crop_tank(
    image,
    detector,
    candidate_labels=None,
    threshold=0.25
):
    """Run detection, produce the annotated preview, and crop the best region."""
    results = detect_tank_region(
        image=image,
        detector=detector,
        candidate_labels=candidate_labels,
        threshold=threshold
    )

    annotated_image = draw_detection_boxes(
        image=image,
        results=results
    )

    crop, best_detection = crop_best_detection(
        image=image,
        results=results
    )

    return {
        "results": results,
        "annotated_image": annotated_image,
        "crop": crop,
        "best_detection": best_detection
    }
