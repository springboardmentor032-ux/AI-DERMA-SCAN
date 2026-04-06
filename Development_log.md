# 📘 DermalScan — Development Log  
### AI Facial Skin Aging Detection System

---

## 📌 Project Overview
DermalScan is a deep learning-based system designed to detect facial skin aging signs such as wrinkles, dark spots, puffy eyes, and clear skin using a pretrained **EfficientNetB0** model. The system includes a web interface for image upload and a backend inference pipeline, forming a complete end-to-end ML workflow.

---

## Development Timeline / Phases

---

### Phase 1: Dataset Setup and Labeling
**TASK:** Collect, inspect, and label dataset into four classes: wrinkles, dark spots, puffy eyes, clear skin  
**RESULT:** Dataset cleaned, labeled, and balanced successfully.

---

### Phase 2: Image Preprocessing & Augmentation
**TASK:** Resize images (224×224), normalize pixel values, apply augmentation (flip, rotation, zoom), and perform one-hot encoding of labels  
**RESULT:** Dataset fully preprocessed and ready for training.

---

### Phase 3: Model Training (EfficientNetB0)
**TASK:** Train a pretrained EfficientNetB0 model using categorical cross-entropy loss and Adam optimizer while tracking performance metrics  
**RESULT:** Model trained successfully with ~98% test accuracy and stable validation performance.

---

### Phase 4: Face Detection Experiment (Haar Cascade)
**TASK:** Use OpenCV Haar Cascade for face detection and cropping before classification  
**RESULT:** Approach was tested but discarded due to inconsistent detection and reduced model performance.

---

### Phase 5: Web UI Development
**TASK:** Develop frontend interface for image upload and prediction visualization using HTML and CSS  
**RESULT:** Completed. UI allows image upload and displays prediction output.

---

### Phase 6: Backend Inference Pipeline
**TASK:** Build backend system for loading trained model and performing inference on uploaded images  
**RESULT:** Completed. Backend successfully integrated model loading and prediction pipeline.

---

### Phase 7: Testing & Generalization
**TASK:** Evaluate model on unseen real-world images to test generalization ability  
**RESULT:** Completed. Model showed strong generalization with consistent predictions across diverse inputs.

---

### Phase 8: Documentation & Final Presentation
**TASK:** Prepare GitHub documentation, user/developer guides, and presentation materials  
**RESULT:** completed implementation.

---

## Tech Stack
- Python 🐍  
- TensorFlow / Keras 🤖  
- OpenCV 👁️  
- HTML + CSS 🌐  
- Jupyter Notebooks 📓  

---

## Current Status
- End-to-end ML pipeline completed  
- Model trained, tested, and validated successfully  
- Web UI + backend integration completed  
- Documentation phase completed  

---