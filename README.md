# Real-Time-Blood-Cell-Detection-Using-Custom-CNN-with-Grad-CAM-Visualization
As an AI &amp; Robotics Engineer, I've been fascinated by the intersection of deep learning and medical diagnostics. So I built HemaVision AI — a full-stack intelligent web platform that classifies white blood cells in real time and explains why it made that decision.


**What it does:**
The system takes a microscopy image of a blood smear and instantly identifies which of the 4 major white blood cell types is present:

🔴 **Eosinophil** — elevated in allergies & parasitic infections 

🟢 **Lymphocyte** — B-cells & T-cells, core of adaptive immunity  

🟡 **Monocyte** — precursors to macrophages, key phagocytic cells  

🔵 **Neutrophil** — first responders, the most abundant WBC type

**What makes it different:**

✅ **99.1% classification accuracy** — trained on thousands of labeled blood cell microscopy images  

✅ **GradCAM Activation Heatmaps** — the AI shows exactly which region of the cell triggered the prediction. No black box. Full explainability.  

✅ **Real-time confidence scores** — animated probability bars for all 4 classes  

✅ **Clinical context** — every prediction comes with normal reference ranges and clinical significance  

✅ **Designed for doctors AND engineers** — clean, cinematic interface built for professional use

**Under the hood:**

- Custom **SeparableConv2D** architecture (9 conv blocks, 69 weight tensors)

- **GlobalAveragePooling** + dual Dense(1024) head

- Trained with **MobileNetV2 preprocessing** + SGD optimizer

- **GradCAM** implementation for gradient-based visual explanations

- **Flask** REST API backend + animated HTML/CSS/JS frontend

- Cross-version Keras compatibility layer (Keras 2 ↔ Keras 3)


This project taught me that building AI for healthcare isn't just about accuracy — it's about **trust**. Clinicians need to understand *why* a model made a decision, not just *what* it decided. GradCAM bridges that gap.

The hematology workflow of tomorrow won't replace pathologists. It will give them AI co-pilots that flag, explain, and accelerate diagnosis.

🔗 **GitHub:** [link in comments]

If you're working in medical AI, computer vision, or healthcare technology — I'd love to connect and hear your thoughts.

#MedicalAI #ComputerVision #DeepLearning #Hematology #HealthcareAI #MachineLearning #GradCAM #ArtificialIntelligence #Bioinformatics #AIinHealthcare #TensorFlow #MedTech #BloodCellClassification #ExplainableAI #XAI

---
---

## 🐙 GITHUB README

---

# 🔬 HemaVision AI — Blood Cell Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-99.1%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.13-orange?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/Flask-2.3-blue?style=for-the-badge&logo=flask"/>
  <img src="https://img.shields.io/badge/GradCAM-Enabled-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Classes-4-purple?style=for-the-badge"/>
</p>

<p align="center">
  <b>A production-ready AI web application for real-time white blood cell classification with explainable GradCAM heatmaps.</b><br/>
  Built for medical professionals and AI engineers.
</p>

---

## 🎯 Overview

HemaVision AI is a full-stack deep learning platform that classifies white blood cells from microscopy images into 4 clinically relevant categories with **99.1% accuracy**. Beyond classification, it generates **GradCAM activation heatmaps** that reveal exactly which morphological features drove the model's decision — making it explainable and trustworthy for clinical use.

---

## 🧬 Supported Cell Types

| Cell Type | Normal Range | Clinical Significance |
|-----------|-------------|----------------------|
| 🔴 **Eosinophil** | 1–4% WBC | Elevated in allergies, asthma, parasitic infections |
| 🟢 **Lymphocyte** | 20–40% WBC | Adaptive immunity; B-cells & T-cells |
| 🟡 **Monocyte** | 2–8% WBC | Phagocytosis; precursor to macrophages |
| 🔵 **Neutrophil** | 55–70% WBC | First-line bacterial defense; most abundant WBC |

---

## ✨ Features


- **🎯 99.1% Classification Accuracy** on held-out test set

- **🔥 GradCAM Heatmaps** — gradient-weighted class activation maps overlaid on the original image

- **📊 Animated Probability Bars** — real-time confidence scores for all 4 classes

- **🏥 Clinical Context** — normal reference ranges and cell function for each prediction

- **🌐 Full-Stack Web App** — drag-and-drop upload, cinematic dark UI, particle animations

- **⚡ Fast Inference** — ~200ms per image on CPU

- **🔄 Keras Version Compatibility** — works across Keras 2 and Keras 3 environments

---

## 🏗️ Model Architecture

```
Input (224×224×3)
│
├── SeparableConv2D(128, 8×8, stride=3) + BN
├── SeparableConv2D(128, 5×5) + BN
├── SeparableConv2D(256, 3×3) + BN
├── SeparableConv2D(256, 1×1) + BN × 2
├── MaxPool2D → SeparableConv2D(512, 3×3) + BN × 4
│
├── GlobalAveragePooling2D
├── Dense(1024, relu) + Dropout(0.5)
├── Dense(1024, relu) + Dropout(0.5)
└── Dense(4, softmax) → [EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL]
```

**Total weight tensors:** 69  

**Preprocessing:** MobileNetV2 normalization (pixels scaled to [-1, 1])  

**Optimizer:** SGD (lr=0.001)  

**Loss:** Categorical Crossentropy  

**Callbacks:** ModelCheckpoint + EarlyStopping (patience=20)

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/hemavision-ai.git
cd hemavision-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your model weights
```
hemavision-ai/
├── app.py
├── best_model.keras   ← place your trained weights here
├── templates/
│   └── index.html
└── requirements.txt
```

### 4. Run the app
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🧪 How GradCAM Works Here

GradCAM (Gradient-weighted Class Activation Mapping) computes the gradient of the predicted class score with respect to the last convolutional layer's feature maps. These gradients are globally average-pooled to produce importance weights, which are then used to create a weighted combination of the feature maps — highlighting the discriminative regions.

```python
# Core GradCAM computation
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, pred_index]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
```

The heatmap is resized to 224×224, colorized with the jet colormap, and alpha-blended (40% heatmap / 60% original) onto the input image.

---

## 📁 Project Structure

```
hemavision-ai/
├── app.py              # Flask backend — model loading, inference, GradCAM
├── templates/
│   └── index.html      # Animated frontend — upload, results, heatmap display
├── requirements.txt    # Python dependencies
├── convert_model.py    # Utility: convert .keras → .h5 for cross-version compat
└── README.md
```

---

## 📦 Requirements

```
flask>=2.3.0
tensorflow>=2.13.0
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.7.0
h5py>=3.8.0
```

---

## 🩺 Dataset

Trained on the **Blood Cell Images** dataset (Kaggle) containing labeled microscopy images across 4 WBC classes, split 70/20/10 (train/validation/test).

---

## 🔬 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **99.1%** |
| Precision (macro) | **99.0%** |
| Recall (macro) | **99.1%** |
| Input Resolution | 224 × 224 px |
| Inference Time | ~200ms (CPU) |



## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## ⭐ If this project helped you, please give it a star!
