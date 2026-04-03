# DermalScan: AI Facial Skin Aging Detection

A deep learning system for detecting facial aging signs (wrinkles, dark spots, puffy eyes, clear skin) using **EfficientNetB0**, with frontend UI under development for image upload and prediction visualization.

---

## 🔹 Project Modules Overview

1. Dataset Setup and Image Labeling  
2. Image Preprocessing, Augmentation, and One-Hot Encoding  
3. EfficientNetB0-based Image Classification (Training & Evaluation)  
4. Face Detection and Prediction Pipeline (Haar Cascade Experiment)  
5. Web UI for Image Upload and Visualization  
6. Backend Pipeline for Model Inference  
7. Testing, Generalization, and Result Validation 
8. Documentation and Final Presentation  

---

## 📌 Module-wise Progress

> ### Module 1: Dataset Setup and Image Labeling
> **TASK:** Collect, inspect, and label dataset into 4 classes:  
> wrinkles, dark spots, puffy eyes, clear skin  
> **RESULT:** Dataset cleaned, labeled, and verified;  
> balanced class distribution achieved  

> ### Module 2: Image Preprocessing, Augmentation, and One-Hot Encoding
> **TASK:** Resize images to 224x224, normalize pixels,  
> apply augmentation (flip, rotation, zoom), and encode labels  
> **RESULT:** All images preprocessed and augmented;  
> labels one-hot encoded; dataset ready for training  

> ### Module 3: EfficientNetB0-based Image Classification (Training & Evaluation)
> **TASK:** Train pretrained EfficientNetB0 with categorical cross-entropy  
> and Adam optimizer; monitor accuracy/loss metrics  
> **RESULT:** Model trained; achieved ~98% test accuracy;  
> validation stable; training curves plotted  

> ### Module 4: Face Detection and Prediction Pipeline (Haar Cascade Experiment)
> **TASK:** Apply OpenCV Haar Cascade for face detection and crop regions  
> for model input  
> **RESULT:** Haar Cascade detected faces inconsistently;  
> reduced classification accuracy; excluded from final pipeline  

> ### Module 5: Web UI for Image Upload and Visualization
> **TASK:** Design frontend interface using HTML/CSS; implement image upload  
> and prediction preview  
> **RESULT:** Pending implementation  

> ### Module 6: Backend Pipeline for Model Inference
> **TASK:** Modularize inference code to load model and return predictions to UI  
> **RESULT:** Pending implementation  

> ### Module 7: Testing, Generalization, and Result Validation  
> **TASK:** Evaluate the trained model on diverse real-world images beyond the training dataset to test generalization.
> **RESULT:** Model successfully tested on unseen images; demonstrated good generalization ability. Predictions were consistent across different inputs, and results were validated using screenshots.

> ### Module 8: Documentation and Final Presentation  
> **TASK:** Prepare user/developer guides, GitHub repository documentation, and presentation materials  
> **RESULT:** Pending implementation   

---

## 🛠️ Tech Stack
- Python, TensorFlow/Keras  
- OpenCV  
- HTML + CSS (Frontend UI)  
- Jupyter Notebooks for EDA & model training