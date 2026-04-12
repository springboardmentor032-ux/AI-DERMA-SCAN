# DermalScan: AI Facial Skin Aging Detection

## 1. Project Overview
DermalScan is a deep learning-based system designed to detect and classify facial aging signs such as wrinkles, dark spots, puffy eyes, and clear skin.

## 2. Core Features
* *Face Detection*: Uses Haar Cascades to focus on facial regions.
* *Classification*: Employs an EfficientNetB0 model for accurate aging sign detection.
* *Real-time Results*: Displays predictions as percentages with interactive progress bars.
* *Data Export*: Allows downloading CSV reports of analysis logs.

## 3. Tech Stack
* *Image Ops*: OpenCV, NumPy
* *Model*: TensorFlow/Keras, EfficientNetB0
* *Backend*: Python (Modularized Inference)
* *Frontend*: HTML, CSS

## 4. How to Use
1. Run python app.py to start the backend.
2. Upload a facial image to the web interface.
3. View the analysis results and download the CSV report.
