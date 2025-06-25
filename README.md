# CIFAR-10 Object Recognition using ResNet50

## Overview

This project implements a deep learning pipeline for object recognition on the CIFAR-10 dataset using a pre-trained ResNet50 backbone. We perform the following steps:

1. **Data Extraction**: Download and extract CIFAR-10 raw data from Kaggle.
2. **Preprocessing**: Load images, map labels, split into train/test, normalize pixel values.
3. **Baseline Model**: Train a simple dense neural network for baseline accuracy.
4. **Transfer Learning**: Upsample 32×32 images to 256×256 and use ResNet50 (ImageNet weights) without top layers, then fine-tune.
5. **Evaluation**: Plot training curves and report test accuracy.
6. **Prediction**: Provide utility to load any image and predict its CIFAR-10 class.

---

## Repository Structure

```
├── cifar10_resnet50.h5    # Trained Keras model (HDF5 format)
├── README.md              # Project documentation
├── notebooks/             # Jupyter/Colab notebooks
│   └── CIFAR10_ResNet50.ipynb
└── requirements.txt       # Python dependencies
```

---

## Requirements

* Python 3.7+
* TensorFlow 2.x
* Keras
* numpy
* pandas
* scikit-learn
* matplotlib
* Pillow
* py7zr

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## Usage

### On Google Colab

1. Upload this notebook to Colab.
2. Ensure you have a `kaggle.json` in your session or use the Kaggle API to download the CIFAR-10 dataset:

   ```bash
   !pip install kaggle
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !kaggle competitions download -c cifar-10
   ```
3. Run all cells to train and evaluate.
4. Save and download the model:

   ```python
   model.save('/content/cifar10_resnet50.h5')
   from google.colab import files
   files.download('/content/cifar10_resnet50.h5')
   ```

### On Kaggle Notebook

1. Create a new Kaggle notebook and enable GPU in Settings.
2. Add the CIFAR-10 dataset under **Add data** → **Competition** → **CIFAR-10**.
3. Copy the code cells from the Colab notebook, but remove any `!pip install kaggle` or API credential steps.
4. Update paths:

   ```python
   TRAIN_DIR = '/kaggle/input/cifar-10/train'
   CSV_PATH  = '/kaggle/input/cifar-10/trainLabels.csv'
   ```
5. Run all cells.
6. Under the **Output** pane, download `cifar10_model.h5` once training completes.

---

## Training Details

* **Dense baseline**: Two-layer MLP on flattened 32×32 images, \~50% accuracy.
* **ResNet50-based**: Upsampled images to 256×256, fine-tuned the pretrained backbone.
* **Hyperparameters**:

  * Optimizer: RMSprop
  * Learning rate: 2e-5
  * Epochs: 10
  * Batch size: 64

---

## Prediction Utility

To predict a custom image:

```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load and preprocess
img = load_img('path/to/image.png', target_size=(256,256))
arr = img_to_array(img) / 255.0
inp = np.expand_dims(arr, 0)

# Predict
pred = model.predict(inp)
cls = np.argmax(pred)
print('Predicted class:', cls)
```

---
