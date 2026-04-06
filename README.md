# AI Skin Condition Classifier

A comprehensive machine learning project for classifying facial skin conditions using computer vision and deep learning techniques. The system includes data augmentation, face detection with Haar cascades, model training with EfficientNetB0, and a Streamlit web interface for real-time predictions.

## Features

- **Data Augmentation**: Generates diverse training images to improve model robustness
- **Face Detection**: Uses Haar cascade classifiers to automatically detect and crop facial regions
- **Deep Learning Model**: EfficientNetB0 architecture for accurate skin condition classification
- **Web Interface**: Streamlit-based front-end for easy image upload and prediction
- **Multi-Class Classification**: Supports 4 skin conditions: clear skin, dark spots, puffy eyes, wrinkles

## Project Structure

```
├── front_end.py                    # Streamlit web application
├── module1_dataset_setup.py        # Dataset exploration and visualization
├── module2_data_augmentation.py    # Data augmentation pipeline
├── module3_efficientB0.py          # Model training with EfficientNetB0
├── module4_haar_cascade.py         # Advanced training with face detection
├── augmented_images/               # Augmented training dataset
├── module3(p2)_results/            # Trained model and training history
├── sample/                         # Python virtual environment
└── README.md                       # This file
```

## Installation

1. **Clone or download the project**
2. **Create virtual environment**:
   ```bash
   python -m venv sample
   ```
3. **Activate environment**:
   - Windows: `sample\Scripts\activate`
   - Linux/Mac: `source sample/bin/activate`
4. **Install dependencies**:
   ```bash
   pip install streamlit tensorflow pillow opencv-python matplotlib pandas seaborn scikit-learn
   ```

## Usage

### Running the Web Application

1. Activate the virtual environment
2. Run the Streamlit app:
   ```bash
   streamlit run front_end.py
   ```
3. Open the provided local URL in your browser
4. Upload a facial image and get instant skin condition prediction

### Training the Model

1. **Dataset Setup** (Module 1):
   ```bash
   python module1_dataset_setup.py
   ```
   - Explores and visualizes the dataset distribution

2. **Data Augmentation** (Module 2):
   ```bash
   python module2_data_augmentation.py
   ```
   - Generates augmented images for better training

3. **Model Training** (Module 3):
   ```bash
   python module3_efficientB0.py
   ```
   - Trains EfficientNetB0 model with face detection

4. **Advanced Training** (Module 4):
   ```bash
   python module4_haar_cascade.py
   ```
   - Enhanced training pipeline with Haar cascades

## Model Details

- **Architecture**: EfficientNetB0 with custom classification head
- **Input Size**: 224x224 RGB images
- **Classes**: 4 skin conditions
- **Face Detection**: Haar cascade for automatic face cropping
- **Training**: Adam optimizer, early stopping, learning rate reduction

## Results

The trained model achieves high accuracy on facial skin condition classification. Training history and results are saved in the `module3(p2)_results/` directory.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- Streamlit
- PIL/Pillow
- NumPy, Pandas, Matplotlib, Seaborn

## Dataset

The project uses augmented facial images categorized into 4 classes:
- Clear skin
- Dark spots
- Puffy eyes
- Wrinkles

## Contributing

This is an internship project. For improvements or modifications, ensure all dependencies are installed and test changes thoroughly.

## License

This project is for educational purposes.