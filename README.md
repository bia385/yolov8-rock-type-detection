# Real-Time Rock Type Detection Using YOLOv8

## Project Overview

This project implements YOLOv8 to detect and classify
three types of rocks from video data in real time:

- Sedimentary Coal
- Metamorphic Marble
- Igneous Diorite

## Tech Stack

- Python
- YOLOv8
- OpenCV
- Roboflow
- Google Colab

## Dataset

The dataset consists of three rock classes and was prepared
using Roboflow for preprocessing, annotation, and dataset splitting.

## Model

The YOLOv8 model was trained for 100 epochs and evaluated
using precision, recall, F1-score, and mAP.

## Results

- mAP: 97.6%
- Precision: 96.8%
- Recall: 97.0%
- Detection Rate: 100%

## Key Insights

Increasing the dataset size improved model performance,
but the improvement was not linear. Dataset quality and
annotation consistency also played an important role.
