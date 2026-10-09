# Real-Time Rock Type Detection Using YOLOv8

> **Undergraduate Thesis Project (Skripsi)**  
> Department of Physics, Faculty of Science and Technology, Universitas Airlangga (2026)  
> **Author:** Robiatul Isnaini (NIM: 182221006)  
> **Supervisors:** Dr. Ir. Soegianto Soelistiono, M.Si. & Yhosep Gita, S.Si., M.T.

---

## 📌 Project Overview

This research evaluates the implementation of the **YOLOv8** deep learning algorithm for real-time detection and classification of three geological rock types from video data:
- **Sedimentary Coal**
- **Metamorphic Marble**
- **Igneous Diorite**

The primary objective of this thesis is to analyze the impact of dataset size variations (50, 100, 200, 300, 400, and 500 images per experiment) on YOLOv8 model performance, detection stability, and accuracy to determine the optimal dataset threshold.

---

## 🧰 Tech Stack & Tools

- **Programming Language:** Python 
- **Framework & Libraries:** Ultralytics YOLOv8, PyTorch, OpenCV, Matplotlib
- **Dataset Platform:** Roboflow & Kaggle
- **Environment:** Google Colaboratory (NVIDIA Tesla T4 GPU)

---

## 📊 Dataset & Experimental Setup

The base dataset contains images of three rock classes (Coal, Marble, and Diorite). Preprocessing, bounding box annotations, Auto-Orient, and resizing to 640×640 pixels were managed via Roboflow, supplemented with data augmentations (flips, rotations, exposure, blur, and noise).

### Dataset Scaling Benchmark
To systematically evaluate the impact of data volume, models were trained across 6 benchmark dataset sizes:
`50 images` ➔ `100 images` ➔ `200 images` ➔ `300 images` ➔ `400 images` ➔ `500 images`

---

## ⚙️ Model Training Parameters

- **Architecture:** YOLOv8 (Medium / YOLOv8m)
- **Image Input Size:** 640×640 pixels
- **Epochs:** 100 epochs (with Early Stopping patience = 128)
- **Batch Size:** 8
- **Evaluation Metrics:** Precision, Recall, F1-Score, mAP@0.5, FPS, Inference Time, Latency

---

## 📈 Key Findings & Performance Results

### 1. Optimal Dataset Threshold (300 Images Benchmark)
The experimental results demonstrate that performance improves significantly during early dataset scaling, reaching peak efficiency at **300 images**:

| Dataset Size | mAP@0.5 | Precision | Recall | Real-Time Accuracy (Video Test) |
| :---: | :---: | :---: | :---: | :---: |
| 50 | 92.2% | 59.4% | 100.0% | Over-detection (High FP) |
| 100 | 91.1% | 95.3% | 96.3% | Moderate duplicates |
| 200 | 99.5% | 99.0% | 98.3% | High accuracy |
| **300 (Optimal)** | **97.6%** | **96.8%** | **97.0%** | **100% Exact Detection Count** |
| 400 | 92.2% | 94.5% | 89.4% | Reduced precision |
| 500 | 91.1% | 85.2% | 89.3% | Increased duplicates / FP |

### 2. Real-Time Inference Speed
Across all dataset variations, the model maintained consistent real-time processing performance:
- **FPS:** ~20 FPS
- **Inference Time:** ~21–22 ms / frame
- **Latency:** ~51–53 ms

---

## 💡 Key Research Insights

1. **Non-Linear Performance Scaling:** Increasing dataset volume from 50 to 300 images consistently improved detection stability and reduced false positives. However, expanding beyond 300 images (400–500 images) led to slight performance degradation due to noise, misclassifications, and duplicate detections.
2. **Optimal Model:** The **300-dataset configuration** achieved the best real-time detection balance, accurately recognizing all actual rock targets in video testing (4 Coal, 3 Diorite, and 3 Marble) with zero extra false bounding boxes.
3. **Class Complexity:** **Metamorphic Marble** demonstrated the most consistent detection stability due to distinct visual traits, whereas **Sedimentary Coal** showed higher sensitivity to false positives caused by surface texture fragments.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install ultralytics opencv-python roboflow
