# 📊 Results and Visualizations

This folder contains the results and visualizations from the B.Tech project implementation.

## Project Results

### Performance Metrics
- **Accuracy**: 95%
- **Sensitivity**: 90%
- **Specificity**: 97%
- **Precision**: 93%
- **F1-Score**: 91.5%
- **AUC-ROC**: 0.96

## Implementation Screenshots

The following figures from Chapter 6 of the B.Tech project report demonstrate the implementation:

### Fig 6.1: Implementation and Installation
- Shows the MATLAB GUI interface for the mammography detection system
- Dataset loading functionality
- Pre-trained model loading interface

### Fig 6.2: Installation Evaluation
- Training progress visualization
- Model performance metrics display
- Real-time accuracy tracking

### Fig 6.3: Mammogram Values Representation
- Preprocessed mammography images
- Feature extraction visualizations:
  - DWT (Discrete Wavelet Transform) coefficients
  - Filtered images using gradient filters
  - CLAHE enhanced contrast images
- Tissue density classification results

### Fig 6.4: Final Result of Mammography Images
- Classification output display
- Predicted tissue density categories:
  - Fatty tissue
  - Glandular tissue
  - Dense tissue
- Confidence scores for predictions

## Visualization Types

### 1. Training Visualizations
- Training progress plots
- Loss curves
- Accuracy curves
- Validation metrics

### 2. Image Processing Visualizations
- Original mammography images (1024×1024 from MIAS dataset)
- CLAHE enhanced images
- Gaussian filtered images
- Morphologically processed images
- Edge detection results

### 3. Feature Extraction Visualizations
- DWT decomposition (Level 2):
  - Approximation coefficients (cAA)
  - Horizontal details (cAH)
  - Vertical details (cAV)
  - Diagonal details (cAD)
- GLCM texture features
- HOG feature maps

### 4. Classification Results
- Confusion matrix
- ROC curves
- Precision-Recall curves
- Classification accuracy per tissue type

### 5. GUI Screenshots
- Main application interface
- Dataset loading panel
- Image display axes
- Classification results panel
- Training status indicators

## Directory Structure

```
results/
├── README.md (this file)
├── metrics/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_history.png
│   └── performance_metrics.csv
├── visualizations/
│   ├── preprocessing/
│   │   ├── clahe_enhancement.png
│   │   ├── noise_reduction.png
│   │   └── morphological_ops.png
│   ├── feature_extraction/
│   │   ├── dwt_decomposition.png
│   │   ├── glcm_features.png
│   │   └── hog_features.png
│   └── classification/
│       ├── predicted_vs_actual.png
│       └── confidence_scores.png
└── gui_screenshots/
    ├── main_interface.png
    ├── dataset_loaded.png
    ├── training_progress.png
    └── classification_output.png
```

## Notes

- All visualizations are generated from the MATLAB implementation (see `matlab/gui.m`)
- Results correspond to the B.Tech project report Chapter 6
- Figures demonstrate 95% accuracy on MIAS dataset classification
- GUI screenshots show the interactive classification interface

## How to Generate Results

### Using MATLAB GUI:
1. Run `matlab/gui.m` in MATLAB
2. Load MIAS dataset using "Load Dataset" button
3. Load pre-trained GoogleNet model
4. Train the network or load saved weights
5. Select test image and classify
6. Screenshots can be captured from the GUI axes

### Using Python Implementation:
```python
from src.train import train_model, evaluate_model
from src.preprocessing import MammographyPreprocessor
from src.models import build_googlenet
import matplotlib.pyplot as plt

# Train and evaluate
model, history = train_model()
metrics = evaluate_model(model, test_data)

# Plot results
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.savefig('results/metrics/training_history.png')
```

## Conference Presentation

These results were presented at:
- **Conference**: 13th International Conference on Science and Innovative Engineering (ICSIE) 2023
- **Date**: 14th May 2023
- **Venue**: Chennai, Tamil Nadu, India
- **Team**: CH. Venkat Sai Ram, A. Ajay, Y. Ganesh
- **Guide**: Ms. M.S. Sivapriya

## References

See the main README.md for academic references and dataset citation.
