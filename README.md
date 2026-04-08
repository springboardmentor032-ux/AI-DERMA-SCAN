DermalScan AI

An AI-powered facial skin analysis system that detects skin conditions like wrinkles, dark spots, puffy eyes, and clear skin using deep learning and provides personalized skincare recommendations.

🚀 Features
📸 Upload facial image
🧠 AI-based skin condition detection
📊 Confidence score visualization
💡 Personalized skincare tips
🎨 Modern animated UI using Streamlit
🧠 Technologies Used
Python
TensorFlow / Keras
OpenCV
Streamlit
NumPy & Pandas
📂 Project Structure
AI_DERMAL/
│
├── train_model.py          # Model training script
├── evaluate_model.py       # Model evaluation script
├── realtime_detection.py   # Haar cascade face detection
├── split_dataset.py        # Dataset splitting
├── utils.py                # Prediction logic
├── app.py                  # Streamlit frontend
├── best_model.keras        # Saved trained model
└── data/                   # Dataset (train/val/test)
⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/AI-DERMA-SCAN.git
cd AI-DERMA-SCAN
2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3. Install dependencies
pip install tensorflow opencv-python streamlit numpy pandas
▶️ Run the Application
cd frontend
streamlit run app.py

👉 Open in browser:

http://localhost:8501
🧪 Model Training

To train the model:

python train_model.py
📊 Model Evaluation

To evaluate performance:

python evaluate_model.py
🧠 How It Works
User uploads an image
Haar Cascade detects face
Image is processed and resized
Deep learning model predicts skin condition
Results + confidence scores displayed
Skincare tips are generated
📈 Model Performance
Accuracy: ~93%
Model: EfficientNetB0 (Transfer Learning)
💡 Future Enhancements
📱 Mobile app integration
🌐 Cloud deployment
🧴 Product recommendation system
👩‍⚕️ Dermatologist integration
👩‍💻 Author

Vaishnavi Kesanakurthi

⭐ Conclusion

DermalScan AI combines deep learning + user-friendly interface to provide real-time skin analysis and personalized skincare guidance.
