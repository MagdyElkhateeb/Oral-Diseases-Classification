# Oral Disease Classification

A deep learning project that classifies oral images into six categories using transfer learning, together with a small web application for running predictions on new images.

The best model, a fine-tuned EfficientNetB0, reaches **81.5% accuracy** and a **0.77 macro F1-score** on the evaluation set, and is served through a FastAPI app.

> **Disclaimer:** This project is a decision-support prototype. It is not a medical device and must not be used as a final diagnosis. Always consult a qualified dentist.

---

## Table of Contents

1. [Classes](#classes)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Installation](#installation)
7. [Running the Web App](#running-the-web-app)
8. [API Reference](#api-reference)
9. [Limitations](#limitations)
10. [Future Improvements](#future-improvements)

---

## Classes

The model predicts one of the following six classes:

| Index | Class               |
|-------|---------------------|
| 0     | Calculus            |
| 1     | Caries              |
| 2     | Gingivitis          |
| 3     | Mouth Ulcer         |
| 4     | Tooth Discoloration |
| 5     | hypodontia          |

The order above is alphabetical and matches the order produced by `image_dataset_from_directory` during training. `app.py` relies on this same order, so do not change it.

---

## Project Structure

```
Task/
|-- Models/
|   |-- final_cnn.keras
|   |-- final_EfficientNetB0.keras
|   |-- final_unfreeze_EfficientNetB0.keras   <- model used by the app
|   |-- final_ResNet50.keras
|   `-- final_unfreeze_ResNet50.keras
|-- Test Images/                              <- one sample image per class
|-- app.py                                    <- FastAPI web app
|-- oral-diseases-classification.ipynb        <- training and evaluation notebook
|-- Model_Comparison_Report.pdf               <- comparison of all five models
`-- README.md
```

---

## Dataset

- Source: the **Oral Diseases** dataset on Kaggle (`salmansajid05/oral-diseases`).
- The notebook collects the original (non-augmented) images of each class into a single clean folder, one sub-folder per class.
- Evaluation set size: **1,112 images**.
- The classes are strongly imbalanced:

| Class               | Images |
|---------------------|-------:|
| Gingivitis          | 482    |
| hypodontia          | 257    |
| Calculus            | 250    |
| Mouth Ulcer         | 53     |
| Caries              | 42     |
| Tooth Discoloration | 28     |

---

## Methodology

### Data pipeline

- Images are loaded with `image_dataset_from_directory` at **224 x 224** resolution, batch size 32.
- 80/20 train/validation split (`validation_split=0.2`, `seed=0`).
- Datasets are cached and prefetched for speed.

### Data augmentation

Applied inside the model during training only:

- Random rotation (0.08)
- Random zoom (plus or minus 10%)
- Random translation (10%)
- Random horizontal flip
- Random contrast and brightness (0.1)
- Gaussian noise (stddev 0.05)

### Models

Five models were trained and compared:

| Model                     | Description                                                              |
|---------------------------|--------------------------------------------------------------------------|
| Baseline CNN              | VGG-style CNN trained from scratch (5 conv blocks, 2 dense layers of 4096) |
| EfficientNetB0            | ImageNet weights, frozen base, new classification head                   |
| EfficientNetB0 + unfreeze | Last 20 layers unfrozen (BatchNorm layers kept frozen), fine-tuned       |
| ResNet50                  | ImageNet weights, frozen base, new classification head                   |
| ResNet50 + unfreeze       | Last 25 layers unfrozen (BatchNorm layers kept frozen), fine-tuned       |

The transfer learning models share the same classification head: Global Average Pooling, Batch Normalization, Dropout (0.4) and a 6-way softmax Dense layer.

### Training setup

- Loss: sparse categorical crossentropy.
- Stage 1 (frozen base): Adam, learning rate `1e-3`.
- Stage 2 (fine-tuning): Adam, learning rate `1e-5`.
- Callbacks: `ReduceLROnPlateau` (factor 0.1, patience 10), `EarlyStopping` (patience 20) and `ModelCheckpoint`, all monitoring validation accuracy.
- Maximum epochs: 200.

---

## Results

### Overall

| Model                     | Accuracy | Macro F1 | Weighted F1 |
|---------------------------|---------:|---------:|------------:|
| Baseline CNN              | 0.4335   | 0.10     | 0.26        |
| EfficientNetB0            | 0.7950   | 0.74     | 0.79        |
| **EfficientNetB0 + unfreeze** | **0.8147** | **0.77** | **0.81** |
| ResNet50                  | 0.7581   | 0.71     | 0.76        |
| ResNet50 + unfreeze       | 0.7977   | 0.76     | 0.80        |

### Per-class F1-score

| Class (n)               | CNN  | Eff  | Eff + unf | Res  | Res + unf |
|-------------------------|-----:|-----:|----------:|-----:|----------:|
| Calculus (250)          | 0.00 | 0.62 | 0.64      | 0.63 | 0.67      |
| Caries (42)             | 0.00 | 0.56 | 0.65      | 0.47 | 0.65      |
| Gingivitis (482)        | 0.60 | 0.81 | 0.83      | 0.75 | 0.80      |
| Mouth Ulcer (53)        | 0.00 | 0.93 | 0.95      | 0.92 | 0.94      |
| Tooth Discoloration (28)| 0.00 | 0.61 | 0.60      | 0.55 | 0.58      |
| hypodontia (257)        | 0.00 | 0.95 | 0.96      | 0.95 | 0.95      |

### Key findings

- The baseline CNN did not learn: it predicts Gingivitis for every image, so its 43% accuracy simply equals the share of Gingivitis in the data.
- Transfer learning with ImageNet weights raised accuracy from 43% to between 76% and 81%.
- Fine-tuning (unfreezing the last layers) improved both architectures: EfficientNetB0 from 79.5% to 81.5%, and ResNet50 from 75.8% to 79.8%.
- Mouth Ulcer and hypodontia are easy for all trained models (F1 between 0.92 and 0.96).
- Calculus, Caries and Tooth Discoloration are the hardest classes (recall of the best model: 0.58, 0.60 and 0.54).

### Model selected for deployment

**EfficientNetB0 + unfreeze** is used in the app. It has the best accuracy, macro F1 and weighted F1, and it is much lighter than ResNet50 (about 5M vs 25M parameters), which makes inference faster and deployment easier. ResNet50 + unfreeze is a close second.

---

## Installation

### Requirements

- Python 3.9 or newer
- TensorFlow / Keras
- FastAPI, Uvicorn, python-multipart
- Pillow, NumPy

### Setup

```bash
# 1. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install tensorflow fastapi uvicorn python-multipart pillow numpy
```

Make sure the trained model file is located at:

```
Models/final_unfreeze_EfficientNetB0.keras
```

The `Models` folder must be next to `app.py`.

---

## Running the Web App

From the project folder, run:

```bash
python app.py
```

Then open <http://127.0.0.1:8000> in your browser.

1. Click **Choose File** and select an oral image (sample images are in `Test Images/`).
2. Click **Predict**.
3. The page shows the predicted class, its confidence, and a probability bar for each of the six classes.

Interactive API documentation is also available at <http://127.0.0.1:8000/docs>.

---

## API Reference

### `GET /`

Returns the HTML interface.

### `POST /predict`

Accepts an image file as `multipart/form-data` under the field name `file`.

**Example request**

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@Test Images/Caries.jpg"
```

**Example response**

```json
{
  "prediction": "Caries",
  "confidence": 87.42,
  "probabilities": {
    "Calculus": 3.10,
    "Caries": 87.42,
    "Gingivitis": 5.01,
    "Mouth Ulcer": 0.85,
    "Tooth Discoloration": 2.97,
    "hypodontia": 0.65
  }
}
```

The values above are illustrative only.

### Preprocessing notes

- The image is converted to RGB and resized to 224 x 224.
- Pixel values are kept in the 0-255 range, because EfficientNet performs its own normalization inside the model. Do not divide by 255 before inference.

---

## Limitations

- The reported scores come from the 1,112-image evaluation set. If this set was also used for checkpointing and early stopping, the scores are slightly optimistic. A separate, untouched test set is needed for an unbiased estimate.
- The gap between the top two models (about 1.7 points) is small and is not a strong difference on its own.
- Recall on the rare classes (54% to 60%) is still low, so the model can miss real cases of Calculus, Caries and Tooth Discoloration.
- The dataset is small and imbalanced, and images may differ from real clinical photos in lighting, angle and quality.

---

## Future Improvements

- Collect more images for the rare classes.
- Use class weights or oversampling to handle the imbalance.
- Add a confidence threshold so that uncertain predictions are flagged for human review.
- Evaluate on a separate held-out test set.
- Add explainability (for example Grad-CAM) to show which image regions drive each prediction.
