import cv2
import numpy as np
from transformers import pipeline


DEFAULT_TANK_LABELS = [
    "large circular metal tank opening",
    "steel cylinder opening",
    "round metal shell",
    "metal tank rim",
]


def load_detector():
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
    if candidate_labels is None:
        candidate_labels = DEFAULT_TANK_LABELS

    results = detector(
        image,
        candidate_labels=candidate_labels,
        threshold=threshold
    )

    return results


def draw_detection_boxes(image, results):
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