# Roll Forming CV Analyzer

Computer vision and process-analysis tool for evaluating rolled shells, tank openings, and cylinder geometry. The application automatically detects a tank opening from an image, extracts the rim profile, quantifies curvature variation, and generates force-correction guidance for roll-forming operators.

---

## Features

### Analysis Model (Computer Vision)

- Automatic tank detection using GroundingDINO
- Automatic image cropping
- Edge extraction using Canny edge detection
- Multi-start radial rim detection
- Circular smoothing of detected geometry
- Curvature error analysis
- Force correction estimation
- Ovality and shape visualization
- Curvature correction zone mapping

### Prediction Model

- Radius sweep analysis
- Visualization of weld effects
- Roll compensation map generation

---

## Analysis Workflow

### 1. Tank Detection

The uploaded image is processed using GroundingDINO to locate the tank opening.

Output:

- Bounding box around detected opening
- Cropped region used for analysis

---

### 2. Edge Detection

The cropped image is converted to grayscale and filtered using a Gaussian blur.

A Canny edge detector identifies the shell boundary.

RGB Image <br>
    ↓ <br>
Grayscale <br>
    ↓ <br>
Gaussian Blur <br>
    ↓ <br>
Canny Edge Detection <br>
    ↓ <br>
Edge Map 

---

### 3. Multi-Start Rim Detection

The algorithm searches radially from the center of the opening.

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

1. Search along the radial line.
2. Find edge candidates.
3. Select the candidate closest to the previous valid radius.
4. Reject unrealistic jumps.
5. Repeat around the circumference.

The median of all runs is used as the final detected rim.

---

### 4. Circular Smoothing

The detected radius profile is smoothed using a circular moving-average filter.

Detected Radius Profile <br>
          ↓ <br>
Circular Moving Average <br>
          ↓ <br>
Smoothed Radius Profile

Allowing for reduced noise while preserving large-scale shape variation.

---

### 5. Curvature Analysis

The detected radius is converted into local curvature.

Curvature is defined as:

κ = 1 / R

where:

- κ = curvature
- R = local radius

<br>

**Higher curvature corresponds to a smaller radius.*

**Lower curvature corresponds to a larger radius.*

---

### 6. Curvature Error Calculation

A target curvature is computed from the desired radius.

κ<sub>target</sub> = 1 / R<sub>target</sub>

<strong> Measured curvature: </strong>

κ<sub>actual</sub> = 1 / R<sub>actual</sub>

<br>

<strong> Curvature Error (%) </strong> = [(κ<sub>actual</sub> − κ<sub>target</sub>)
/ κ<sub>target</sub> ] × 100

Interpretation:

- Positive error → shell is too tight
- Negative error → shell is too flat

---

### 7. Force Correction Estimation

The required force correction is estimated from curvature mismatch.

<strong>Force Correction (%) </strong> = [(κ<sub>target</sub> − κ<sub>actual</sub>)
 / κ<sub>actual</sub>] × 100

Interpretation:

- Positive correction → increase bending force
- Negative correction → decrease bending force

---

### 8. Correction Zones

Each point around the shell is classified:

| Zone | Meaning |
|--------|---------|
| Green | Within tolerance |
| Blue | Too flat – increase force |
| Red | Too tight – decrease force |

This creates a visual correction map for operators.


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

## Requirements
```text
streamlit>=1.45
opencv-python>=4.10
numpy>=1.26
matplotlib>=3.8
Pillow>=10.0
transformers>=4.50
torch>=2.7
torchvision>=0.22
```

---


## Run

```bash
pip install -r requirements.txt
streamlit run main.py
```

---

## Future Development

- Automatic weld seam detection
- Roller-gap compensation estimation
- Camera-guided live measurements
- FEA integration
- Historical production tracking
- Automatic machine parameter recommendations
- Digital integration for roll-forming systems