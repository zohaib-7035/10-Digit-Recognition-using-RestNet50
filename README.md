
# 🧠 CIFAR-10 Object Recognition using ResNet50

This project is a deep learning pipeline to classify 10 categories of images from the CIFAR-10 dataset using a ResNet50-based model. It also includes a basic dense baseline and steps for deploying the trained model via Streamlit.

---

## 📦 Dataset

- **Dataset Source:** [CIFAR-10 - Object Recognition in Images (Kaggle)](https://www.kaggle.com/c/cifar-10)
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Format:** PNG images + CSV labels

---

## 🏗️ Project Structure

```

📁 cifar10-project/
├── cifar10\_model.h5         # Trained ResNet50 model
├── model\_files.zip          # (Optional) zipped export of model + outputs
├── streamlit\_app.py         # Streamlit app for deployment
├── train\_notebook.ipynb     # Kaggle or Colab training notebook
└── README.md                # You're reading it

````

---

## 🚀 Model Architectures

### 🔹 1. Simple Dense Baseline
- Input: (32, 32, 3)
- Flatten → Dense(64, ReLU) → Dense(10, Softmax)

### 🔹 2. ResNet50 Transfer Learning
- Input: (32, 32, 3) → Upsampled to (256, 256, 3)
- Pretrained ResNet50 base (imagenet weights, `include_top=False`)
- Custom classifier:
  - Flatten → BatchNorm → Dense(128, ReLU) → Dropout
  - BatchNorm → Dense(64, ReLU) → Dropout
  - BatchNorm → Dense(10, Softmax)

---

## 🛠️ Dependencies

```bash
pip install tensorflow keras numpy pandas matplotlib pillow streamlit
````

---

## 🧪 Training

```python
# Data preprocessing
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# Training
model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=10)
```

---

## 💾 Saving the Model

```python
model.save("cifar10_model.h5")
```

To download in Kaggle:

```python
from IPython.display import FileLink
FileLink("/kaggle/working/cifar10_model.h5")
```

---

## 🧠 Testing with a Single Image

```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("cifar10_model.h5")
img = Image.open("test_image.png").convert('RGB')
img = img.resize((256, 256))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
```

---

## 🌐 Streamlit Deployment

Create a file `streamlit_app.py`:

```python
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("cifar10_model.h5")

label_map = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

st.title("CIFAR-10 Object Classifier")

uploaded_file = st.file_uploader("Upload a PNG Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((256, 256))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = label_map[np.argmax(prediction)]
    st.write(f"### 🎯 Prediction: **{predicted_class}**")
```

Run it with:

```bash
streamlit run streamlit_app.py
```

---

## ✅ Results & Accuracy

* Accuracy after 10 epochs: \~75-85% (depending on training parameters and augmentations)
* Loss: Reduced steadily over training

---

## 📌 Notes

* Ensure image shape is `(256, 256, 3)` before prediction.
* If deploying on Streamlit, upload the `.h5` model and test images.
* For faster deployment, reduce ResNet layers or switch to MobileNetV2.

---

## 🧑‍💻 Author

**Zohaib Shahid**
Feel free to contribute or fork this repository.

```

---

