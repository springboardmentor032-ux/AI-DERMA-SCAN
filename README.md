🧴 DermalScan AI

✨ AI-Powered Facial Skin Analysis System

💖 About the Project

DermalScan AI is a smart web-based application that uses Artificial Intelligence and Computer Vision to analyze facial skin conditions.

By simply uploading an image, the system can:

🔍 Detect the face using Haar Cascade
🧠 Classify skin condition using Deep Learning
📊 Show prediction confidence
💡 Provide personalized skincare suggestions

👉 It acts like a virtual skin assistant, helping users understand their skin health instantly.

🎯 Problem Statement

Skin issues like wrinkles, dark spots, and puffy eyes are common, but early detection is difficult without professional consultation.

❗ Challenges:
Lack of awareness about skin conditions
No quick and accessible diagnostic tools
Cost of dermatological consultations
✅ Solution:

DermalScan AI provides a low-cost, fast, and intelligent solution using AI to analyze skin conditions directly from images.

🌟 Features
📸 Upload facial image easily
🧠 AI-based classification (4 categories)
🟢 Face detection using OpenCV Haar Cascade
📊 Confidence score visualization (bar chart)
💡 Smart skincare recommendations
🎨 Modern and interactive UI using Streamlit
⚡ Fast processing (< 5 seconds per image)
🔍 Detection Output
4
📊 Confidence Analysis
4
📉 Confusion Matrix
5
🧠 Technologies Used
Category	Tools
Programming	Python
Deep Learning	TensorFlow, Keras
Computer Vision	OpenCV
Frontend	Streamlit
Data Handling	NumPy, Pandas
Visualization	Matplotlib
🏗️ System Architecture
📥 Image Input (User Upload)
👁️ Face Detection (Haar Cascade)
🔄 Image Preprocessing (Resize, Normalize)
🧠 Model Prediction (EfficientNetB0)
📊 Output Visualization
💡 Recommendation Engine
📂 Project Structure
AI_DERMAL/
│
├── train_model.py          # Model training
├── evaluate_model.py       # Model evaluation
├── realtime_detection.py   # Face detection
├── split_dataset.py        # Dataset preparation
├── utils.py                # Prediction logic
│
├── frontend/
│   ├── app.py              # UI (Streamlit)
│   └── utils.py            # Inference pipeline
│
├── dermalscan_model.keras  # Trained model
└── data/                   # Dataset
⚙️ Installation
git clone https://github.com/your-username/AI-DERMA-SCAN.git
cd AI-DERMA-SCAN

python -m venv venv
venv\Scripts\activate

pip install tensorflow opencv-python streamlit numpy pandas matplotlib
▶️ Run the Application
cd frontend
streamlit run app.py

👉 Open in browser:

http://localhost:8501
🧪 Model Training
Uses EfficientNetB0 (Pretrained Model)
Transfer Learning applied
Data Augmentation used:
Rotation
Zoom
Horizontal Flip
python train_model.py
📊 Model Evaluation

Evaluation includes:

Accuracy & Loss
Confusion Matrix
Classification Report
python evaluate_model.py
📈 Model Performance
Metric	Value
Accuracy	~93%
Loss	Low
Model	EfficientNetB0
Classes	4
🧠 How It Works
User uploads image
Face is detected using Haar Cascade
Image is resized to 224x224
Model predicts skin condition
Confidence scores are displayed
Skincare tips are generated
💄 Skincare Recommendation Logic
Condition	Recommendation
Wrinkles	Retinol, hydration, sunscreen
Dark Spots	Vitamin C, Niacinamide
Puffy Eyes	Eye cream, sleep improvement
Clear Skin	Maintain skincare routine
🔮 Future Enhancements
📱 Mobile application
🌐 Cloud deployment
🧴 Product recommendation system
🤖 AI chatbot for skincare advice
🧬 Detection of more skin conditions
👩‍💻 Author

Vaishnavi Kesanakurthi

⭐ Conclusion

DermalScan AI successfully combines:

🧠 Deep Learning
👁️ Computer Vision
💻 Interactive UI

to create a real-time intelligent skincare analysis system.
