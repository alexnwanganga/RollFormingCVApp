# Roll Forming CV Analyzer

Computer vision and process-analysis tool for evaluating rolled shells, tank openings, and cylinder geometry. The application detects a tank opening from an uploaded image, extracts the rim profile, quantifies curvature variation around the circumference, and generates roll-force correction guidance for operators and process engineers.

This project is intended to help answer a practical shop-floor question:

> Where is the rolled shell too flat, where is it too tight, and how should forming force be adjusted around the circumference?

The app is a guidance tool. It is not yet a calibrated machine-control or force-prediction system. Any production use should be validated against known measurements, operator feedback, and machine/process data.

---

## Problem Statement

Rolled shells and large cylindrical parts can come out of forming with local shape errors:

- Sections with a radius that is too large are too flat.
- Sections with a radius that is too small are too tight or over-bent.
- Weld seams, lighting, handling distortion, and camera angle can make manual inspection difficult.
- Operators need a fast way to identify where correction is needed around the rim.

Traditional measurement can be slow, manual, and hard to visualize. This application uses image processing to estimate the rim profile from a photo and turn that profile into curvature and correction-zone views.

---

## Solution Overview

The application runs as a Streamlit dashboard.

At a high level:

1. User uploads a photo of a rolled shell / tank opening.
2. GroundingDINO attempts to detect and crop the tank region.
3. OpenCV preprocessing converts the crop into an edge map.
4. A multi-start radial search extracts a closed rim profile.
5. The radius profile is smoothed around the full circumference.
6. Curvature error is calculated from the detected local radius.
7. Correction zones are visualized on the image.
8. A harmonic rim equation is fitted and exported as JSON or CSV.

Image <br>
    ↓ <br>
Detection / Crop <br>
    ↓ <br>
Edge Map <br>
    ↓ <br>
Rim Profile <br>
    ↓ <br>
Curvature Analysis <br>
    ↓ <br>
Correction Guidance + Equation Export

---

## Features

### Analysis Model (Computer Vision)

- Automatic tank detection using GroundingDINO
- Automatic image cropping
- Manual geometry override when detection is imperfect
- Edge extraction using grayscale conversion, Gaussian blur, and Canny edge detection
- Multi-start radial rim detection
- Circular smoothing of detected geometry
- Curvature error analysis
- Force correction estimation
- Correction zone mapping directly over the uploaded image
- Derived rim equation fitting and visualization
- JSON and CSV export of rim-equation coefficients

### Operator / Engineering Dashboard

- Upload image workflow
- Sidebar controls for detection, edge detection, rim search, and curvature settings
- Expanders for uploaded image, detected region, analysis crop, and edge map
- Summary metrics for target radius, tolerance status, and pixels-per-inch scale
- Largest force increase/decrease callouts
- Plots for curvature error, force correction, and derived rim equation

### Future / Experimental Functionality

- Camera capture helper exists in `src/camera_capture.py`, but it is not currently wired into the main upload workflow.
- Prediction-model ideas are listed under Future Development. They are not fully implemented as a production model in the current codebase.

---

## Analysis Workflow

### 1. Image Upload

The user uploads a `.jpg`, `.jpeg`, or `.png` image through the Streamlit UI.

Recommended image conditions:

- The rim should be clearly visible.
- Backgrounds should be as simple as possible.
- The camera should be roughly centered on the opening.
- The opening should occupy a meaningful portion of the image.
- Strong glare, shadows, occlusions, or clutter can reduce detection quality.

Output:

- PIL RGB image
- Raw uploaded bytes used for Streamlit caching

---

### 2. Tank Detection

The uploaded image is processed using GroundingDINO through the Hugging Face `transformers` pipeline.

Default candidate labels:

- `large circular metal tank opening`
- `steel cylinder opening`
- `round metal shell`
- `metal tank rim`

Output:

- Bounding box around the detected opening
- Annotated preview image
- Cropped region used for analysis
- Best detection metadata

If no detection is found, the app falls back to the full uploaded image. In that case, the manual geometry controls become more important.

---

### 3. Edge Detection

The cropped image is converted to grayscale and filtered using a Gaussian blur.

A Canny edge detector identifies candidate shell boundaries.

RGB Image <br>
    ↓ <br>
Grayscale <br>
    ↓ <br>
Gaussian Blur <br>
    ↓ <br>
Canny Edge Detection <br>
    ↓ <br>
Edge Map

Relevant sidebar controls:

| Setting | Purpose |
|--------|---------|
| Low Gradient Threshold | Lower Canny threshold |
| High Gradient Threshold | Upper Canny threshold |
| Blur Kernel Size | Gaussian blur size before edge detection |

Notes:

- Higher blur can reduce noise but may soften the rim edge.
- Lower Canny thresholds find more edges but may include clutter.
- Higher Canny thresholds reduce clutter but may miss weak rim sections.

---

### 4. Approximate Geometry

The app starts with a default center and radius based on the crop dimensions:

- Center X = crop width / 2
- Center Y = crop height / 2
- Expected Radius = half of the smaller crop dimension

The user can override these values in the sidebar.

These controls matter because the rim detector searches radially around the expected circle. If the center or radius is too far off, the radial search can lock onto the wrong edge.

---

### 5. Multi-Start Rim Detection

The algorithm searches radially from the approximate center of the opening.

Instead of using a single starting angle, multiple starting angles are evaluated:

0°  
45°  
90°  
135°  
180°  
225°  
270°  
315°

For each angle:

1. Search along the radial line within the configured search band.
2. Find edge candidates from the Canny edge map.
3. Select the candidate closest to the previous valid radius.
4. Reject unrealistic jumps larger than the max step change.
5. Carry the previous radius through missing/noisy edge sections.
6. Repeat around the circumference.

The median of all start-angle runs is used as the final detected rim.

Relevant sidebar controls:

| Setting | Purpose |
|--------|---------|
| Angular Samples | Number of points around the circumference |
| Search Band [pixels] | Distance inside/outside expected radius to scan |
| Max Step Change [pixels] | Maximum allowed radius jump between adjacent samples |
| Smoothing Window | Circular moving-average window |

---

### 6. Circular Smoothing

The detected radius profile is smoothed using a circular moving-average filter.

Detected Radius Profile <br>
          ↓ <br>
Circular Moving Average <br>
          ↓ <br>
Smoothed Radius Profile

The smoothing is circular, meaning the first and last angular samples are treated as neighbors. This avoids a discontinuity at 0 / 2π radians.

Purpose:

- Reduce edge-detection noise
- Preserve large-scale shape variation
- Produce a stable profile for curvature analysis

---

### 7. Curvature Analysis

The detected radius is converted into local curvature.

Curvature is defined as:

κ = 1 / R

where:

- κ = curvature
- R = local radius

<br>

**Higher curvature corresponds to a smaller radius.**

**Lower curvature corresponds to a larger radius.**

Interpretation:

| Condition | Meaning |
|----------|---------|
| Radius too large | Section is too flat |
| Radius too small | Section is too tight |
| Curvature too low | Section needs more bend |
| Curvature too high | Section needs less bend |

---

### 8. Target Radius Modes

The app supports two target-radius modes.

| Mode | Meaning | Best Use |
|------|---------|----------|
| Median detected radius | Uses the median detected rim radius as the target | Evaluating relative variation around the observed part |
| Expected/manual radius | Uses the sidebar expected radius as the target | Comparing the part against a known desired radius |

The known actual radius in inches is used to calculate pixels per inch:

pixels_per_inch = expected_radius_pixels / real_radius_inches

This scale is used for reporting and exported coefficients.

---

### 9. Curvature Error Calculation

A target curvature is computed from the target radius.

κ<sub>target</sub> = 1 / R<sub>target</sub>

<strong>Measured curvature:</strong>

κ<sub>actual</sub> = 1 / R<sub>actual</sub>

<br>

<strong>Curvature Error (%)</strong> = [(κ<sub>actual</sub> − κ<sub>target</sub>)
/ κ<sub>target</sub>] × 100

Interpretation:

- Positive mathematical error → shell is too tight
- Negative mathematical error → shell is too flat

Note:

The dashboard flips some signs so operator-facing plots align with correction guidance. Before changing chart labels or force-correction signs, check both `main.py` and `src/correction.py`.

---

### 10. Force Correction Estimation

The required force correction is estimated from curvature mismatch.

<strong>Force Correction (%)</strong> = [(κ<sub>target</sub> − κ<sub>actual</sub>)
/ κ<sub>actual</sub>] × 100

Interpretation:

- Positive correction → increase bending force
- Negative correction → decrease bending force

Important:

This is a geometry-based heuristic. It does not yet model material properties, springback, roller geometry, weld effects, tooling state, or machine calibration.

---

### 11. Correction Zones

Each point around the shell is classified:

| Zone | Meaning | Operator Guidance |
|--------|---------|------------------|
| Green | Within tolerance | No major correction needed |
| Blue | Too flat | Increase force / add bend |
| Red | Too tight | Decrease force / reduce bend |

This creates a visual correction map for operators.

---

### 12. Derived Rim Equation

After rim detection, the app fits a harmonic equation to the detected radius profile.

Equation form:

R(θ) = R<sub>t</sub> + A<sub>3c</sub>cos(3θ) + A<sub>3s</sub>sin(3θ) + A<sub>0c</sub>cos(2θ) + A<sub>0s</sub>sin(2θ) + A<sub>ec</sub>cos(θ) + A<sub>es</sub>sin(θ) + A<sub>s</sub>

Term interpretation:

| Term | Meaning |
|------|---------|
| R<sub>t</sub> | Target radius |
| 3θ terms | Three-lobed / roll-process variation |
| 2θ terms | Ovality |
| 1θ terms | Egg-shaped error or remaining center bias |
| A<sub>s</sub> | Constant radius offset |

The app also performs a local center refinement before finalizing the fit. This reduces the chance that a small center error appears as false first-harmonic shape error.

Exports:

- `derived_rim_equation.json`
- `derived_rim_coefficients.csv`

The exported coefficients are converted to inches.

---

## Project Structure

```text
RollFormingCVApp/
│
├── main.py
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── detection.py
│   ├── preprocessing.py
│   ├── rim_analysis.py
│   ├── correction.py
│   └── camera_capture.py
│
└── uploads/
```

---

## Module Responsibilities

| File | Responsibility |
|------|----------------|
| `main.py` | Streamlit UI, pipeline orchestration, caching, plotting, export generation |
| `src/detection.py` | GroundingDINO model loading, zero-shot object detection, annotation, crop selection |
| `src/preprocessing.py` | RGB-to-gray conversion, Gaussian blur, Canny edge generation |
| `src/rim_analysis.py` | Multi-start radial rim search, circular smoothing, coordinate generation |
| `src/correction.py` | Pixel/inch scaling, curvature calculations, tolerance classification, force-correction heuristic |
| `src/camera_capture.py` | Optional Streamlit camera-capture component for future live workflows |
| `uploads/` | Sample / local uploaded images used during development |

---

## Runtime Flow

```text
main.py
│
├── get_detector()
│   └── src.detection.load_detector()
│
├── cached_detection()
│   └── src.detection.detect_and_crop_tank()
│
├── cached_edge_map()
│   └── src.preprocessing.generate_edge_map()
│
├── cached_rim_detection()
│   └── src.rim_analysis.detect_rim_multistart()
│
├── compute_curvature_correction()
│   └── src.correction.compute_curvature_correction()
│
└── fit_rim_equation()
    └── local harmonic equation fit in main.py
```

Streamlit reruns `main.py` whenever controls change. Expensive operations are cached:

- Detector model loading uses `st.cache_resource`.
- Detection, edge-map generation, and rim detection use `st.cache_data`.

---

## Requirements

```text
streamlit>=1.45,<2.0
numpy>=1.26,<3.0
matplotlib>=3.8,<4.0
Pillow>=10.0,<12.0
opencv-python-headless>=4.10,<5.0
transformers>=4.50,<5.0
huggingface-hub>=0.24
safetensors>=0.4
torch>=2.3
torchvision>=0.18
```

Notes:

- The first detector run may download the GroundingDINO model from Hugging Face.
- `torch` and `torchvision` can be large dependencies. CPU installs work, but detector inference is faster with a GPU-appropriate PyTorch build.
- `opencv-python-headless` is used for Streamlit/server compatibility. If a future camera workflow needs desktop camera backends, use `opencv-python` instead of installing both OpenCV packages.
- Runtime performance depends heavily on image size, detector speed, and the number of angular samples.

---

## Setup

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Run

```bash
streamlit run main.py
```

Then open the local Streamlit URL shown in the terminal.

---

## How To Use The App

1. Start the Streamlit app.
2. Upload a clear image of the rolled shell / tank opening.
3. Keep auto-crop enabled for the first pass.
4. Inspect the detected tank region.
5. If detection fails, disable auto-crop or adjust the detection threshold.
6. Inspect the edge detection output.
7. Adjust Canny thresholds and blur until the rim edge is clear.
8. Adjust center and expected radius if the rim search is off.
9. Review correction zones and summary metrics.
10. Download JSON or CSV coefficients if downstream analysis is needed.

---

## Tuning Guide

### Detection Threshold

Lower values produce more detections.

Use when:

- The tank is not detected.
- The image is low contrast.
- The detector is too conservative.

Risk:

- False detections may increase.

### Canny Thresholds

Use lower thresholds when the rim edge is weak.

Use higher thresholds when the background produces too many edges.

### Blur Kernel

Use larger blur kernels to reduce noisy texture.

Risk:

- Too much blur can erase thin or low-contrast rim edges.

### Search Band

Increase when the expected radius is approximate or the rim is far from the default circle.

Risk:

- Wider search bands can capture unrelated edges.

### Max Step Change

Increase when the actual rim has abrupt local changes.

Decrease when the detector jumps between unrelated edges.

### Smoothing Window

Increase for a smoother engineering trend.

Decrease to preserve sharper local variation.

---

## Known Assumptions

- The rim is approximately circular and can be represented as radius versus angle.
- The image crop contains the relevant opening.
- The approximate center and radius are close enough for radial search.
- Local curvature is approximated using radius from the estimated center.
- The known actual radius is accurate enough for pixel-to-inch scaling.
- Correction guidance is based on geometry only, not a calibrated forming-force model.

---

## Known Limitations

- Poor lighting, reflections, and cluttered backgrounds can break edge detection.
- Off-axis camera perspective can distort the apparent rim shape.
- The detector may crop the wrong object if the image contains multiple circular metal features.
- The force-correction estimate does not account for material grade, thickness, springback, machine condition, or tooling setup.
- The current camera-capture component is not connected to the main workflow.
- There is no automated test suite yet.

---

## Handoff Notes For Future Engineers

### Most Important Files

- Start with `main.py` to understand the full workflow.
- Read `src/rim_analysis.py` before changing rim extraction behavior.
- Read `src/correction.py` before changing sign conventions, labels, or force guidance.
- Read `src/detection.py` before swapping or tuning the object detector.

### Areas To Be Careful With

- Streamlit cache keys depend on function inputs. If a cached function starts using new hidden state, include that state as an explicit argument.
- The detector result format comes from `transformers`. If the model or pipeline changes, validate the shape of `score`, `label`, and `box`.
- The dashboard flips some signs for operator-facing interpretation. Keep mathematical signs and displayed signs documented together.
- The rim search assumes a continuous edge around the circumference. Changes to jump rejection or smoothing can significantly change correction output.
- Exported rim-equation coefficients are in inches, while most image-processing work is in pixels.

### Suggested Validation After Changes

Run a syntax check:

```bash
python -m py_compile main.py src/camera_capture.py src/correction.py src/detection.py src/preprocessing.py src/rim_analysis.py
```

Manual smoke test:

1. Start the app with `streamlit run main.py`.
2. Upload one of the images in `uploads/`.
3. Confirm the app produces:
   - detected region or full-image fallback
   - edge detection output
   - correction zone plot
   - curvature error plot
   - force correction plot
   - derived rim equation plot
   - JSON and CSV download buttons

Recommended future tests:

- Unit tests for `compute_curvature_correction()`
- Unit tests for `circular_smooth()`
- Synthetic edge-map tests for `detect_rim_multistart()`
- Regression fixtures using known sample images

---

## Future Development

- Automatic weld seam detection
- Roller-gap compensation estimation
- Camera-guided live measurements
- Integrate `src/camera_capture.py` into the main workflow
- FEA integration
- Historical production tracking
- Automatic machine parameter recommendations
- Digital integration for roll-forming systems
- Calibrated correction model using measured machine/process data
- Test suite and sample-image regression checks
- Perspective correction / camera calibration
- Optional segmentation model for more reliable rim extraction

---

## Project Status

Current state:

- Functional Streamlit prototype
- Image upload workflow implemented
- Detection, edge extraction, rim analysis, correction guidance, and exports implemented
- Comments and README updated for engineering handoff

Not yet production-calibrated:

- Force prediction
- Machine parameter recommendation
- Camera-guided measurement workflow
- Automated quality gates / tests
