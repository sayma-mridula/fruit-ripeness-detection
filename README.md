# 🍎 Fruit Ripeness Detection System

A computer vision application that analyzes fruit images to detect and classify ripeness levels using HSV color segmentation and texture analysis — built with Python, OpenCV, and Tkinter.

---

## 📸 Features

- **Automatic fruit segmentation** from background using edge detection and contour analysis
- **Ripeness classification** into three categories: Unripe, Ripe, and Overripe
- **Multi-stage analysis pipeline** combining color segmentation and texture variance
- **Interactive GUI** with tabbed visualization panels
- **Detailed metrics** including saturation, green percentage, dark pixel ratio, and texture variance
- **Visual overlays** showing ripeness regions color-coded on the original image

---

## 🧠 How It Works

The system follows a multi-step image processing pipeline:

```
Input Image
    │
    ▼
HSV Color Conversion + Gaussian Blur
    │
    ▼
Fruit Segmentation (Edge Detection → Contour Masking)
    │
    ▼
Color-Based Ripeness Segmentation
    │   ├── Unripe  → Green hues (40–85° hue)
    │   ├── Ripe    → Red (0–10°, 170–180°) / Yellow (15–50°)
    │   └── Overripe→ Dark pixels + Brown hues (5–35°)
    │
    ▼
Texture Variance Analysis (Refines overripe detection)
    │
    ▼
Final Classification + Visualization
```

---

## 🖥️ GUI Overview

The application has 5 tabbed panels:

| Tab | Description |
|-----|-------------|
| **Original & Result** | Side-by-side view of input image and color-coded analysis result |
| **Processing Steps** | Step-by-step pipeline: HSV conversion, channel splits, edge detection, segmentation |
| **Ripeness Masks** | Individual binary masks for Unripe, Ripe, and Overripe regions |
| **Color Segmentation** | Detailed view of hue range detections (green, red, yellow, dark) |
| **Texture Analysis** | Variance map, high-variance regions, dark regions, and pie chart |

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install opencv-python numpy pillow matplotlib
```

> Tkinter is included with standard Python installations. If missing, install via your OS package manager.

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fruit-ripeness-detection.git
cd fruit-ripeness-detection
```

---

## 🚀 Usage

```bash
python ripeness.py
```

1. Click **"Upload Fruit Image"** to load a fruit photo (`.jpg`, `.png`, `.bmp`, `.tiff`)
2. Click **"🔍 Analyze Ripeness"** to run the detection pipeline
3. View results across the tabbed panels on the right
4. Check the **Analysis Results** panel on the left for classification and percentage breakdown

---

## 📊 Output Metrics

| Metric | Description |
|--------|-------------|
| **Overall Classification** | Unripe / Ripe / Overripe |
| **Unripe %** | Percentage of fruit area classified as unripe |
| **Ripe %** | Percentage of fruit area classified as ripe |
| **Overripe %** | Percentage of fruit area classified as overripe |
| **Average Saturation** | Mean HSV saturation of fruit pixels |
| **Green Percentage** | Fraction of pixels in green hue range |
| **Dark Pixel %** | Fraction of pixels with low value (V < 100) |
| **Average Variance** | Texture roughness of the fruit surface |
| **Texture Overripe %** | Percentage of high-variance/dark regions |

---

## 📁 Project Structure

```
Fruit_Ripeness_detection/
│
├── ripeness.py          # Main application (detector + GUI)
├── Images/              # Sample fruit images for testing
├── Report/              # Project report documents
├── Slide/               # Presentation slides
└── README.md            # This file
```

---

## 🔬 Technical Details

### Color Segmentation Logic

- **Unripe**: Hue 40–85° with saturation 30–190, OR low-saturation light pixels
- **Ripe**: Hue 0–10° or 170–180° (red) with saturation > 70; or Hue 15–50° (yellow) with saturation > 120
- **Overripe**: Value < 100 (dark pixels) OR Hue 5–35° with saturation > 40 (brown)

### Texture Refinement

Texture variance is computed using a local sliding window (15×15 kernel). Regions with high variance combined with dark pixels are reclassified as overripe, improving accuracy for bruised or spoiled fruit.

### Classification Rules (Priority Order)

1. Green % > 40 → **Unripe**
2. Dark pixel % > 8 → **Overripe**
3. High saturation + ripe % > 45 → **Ripe**
4. High variance + overripe indicators → **Overripe**
5. Majority class fallback

---

## 🧪 Supported Fruits

The system works best with:
- 🍌 Bananas (yellow/green)
- 🍎 Apples (red/green)
- 🍊 Oranges
- 🍋 Lemons
- 🍇 Grapes
- 🥭 Mangoes

> Performance may vary for fruits with non-standard coloration or complex backgrounds.

---

## 📷 Sample Results

Color coding on result overlay:
- 🟢 **Green** = Unripe regions
- 🟡 **Yellow** = Ripe regions  
- 🔴 **Red** = Overripe regions

---

## 📄 License

This project is for academic/educational purposes.

---

## 👤 Author

Developed as part of an Image Processing Lab project.

> Feel free to open issues or submit pull requests for improvements!
