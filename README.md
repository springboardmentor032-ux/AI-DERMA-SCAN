DermalScan AI
(AI-Powered Facial Skin Analysis System)



1. Project Overview

DermalScan AI is a web-based intelligent system designed to analyze facial skin conditions using deep learning and computer vision techniques. The application identifies common skin conditions such as wrinkles, dark spots, puffy eyes, and clear skin from user-uploaded facial images.The system integrates image processing and convolutional neural networks to generate predictions along with confidence scores and basic skincare recommendations.



2. Objectives
Develop a deep learning model for skin condition classification
Implement face detection using computer vision
Provide real-time predictions through a web interface
Display confidence scores for predictions
Generate skincare recommendations


 
 3. Problem Statement
Skin-related issues require early detection, but access to dermatological consultation is often limited.

Challenges:
Lack of quick diagnostic tools
High consultation costs
Limited awareness
Solution:

DermalScan AI provides an automated AI-based system for instant skin analysis using images.

4. Features

Image upload functionality
Face detection using Haar Cascade
Deep learning-based classification
Confidence score visualization
Skincare recommendation system
Interactive UI using Streamlit
 
 
 6. Sample Output

The system detects the face, draws a bounding box, and displays the predicted skin condition with confidence percentage.
<img width="512" height="384" alt="image" src="https://github.com/user-attachments/assets/119dfd78-3aa6-4f91-b7ed-18687b43b5dd" />
<img width="885" height="362" alt="image" src="https://github.com/user-attachments/assets/174d7d5e-096e-4fda-b5db-a286f04626a0" />


6. Technologies Used

   
Category	Tools
Programming Language	Python
Deep Learning	TensorFlow, Keras
Computer Vision	OpenCV
Frontend	Streamlit
Data Processing	NumPy, Pandas
Visualization	Matplotlib
 

 8. System Architecture
User uploads image
Face detection using Haar Cascade
Image preprocessing
Feature extraction using EfficientNetB0
Classification
Result visualization
  
  
  9. Dataset Description
Classes:
Wrinkles
Dark Spots
Puffy Eyes
Clear Skin
Data split:
Training
Validation
Testing
Augmentation:
Rotation
Zoom
Flipping
  
  
  10. Model Architecture
EfficientNetB0 (pretrained)
Global Average Pooling
Dense layers
Dropout
Softmax output

11. Training Methodology
Loss: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy
Input size: 224 × 224

Callbacks:

EarlyStopping
ModelCheckpoint
 
 
 12. Model Evaluation
Accuracy and Loss
Confusion Matrix
Classification Report
 
 
 13. Model Performance

      
Metric	Value
Accuracy	~93%
Model	EfficientNetB0
Classes	4
 
 
 14. Working Methodology


Upload image
Detect face
Preprocess image
Predict using model
Display results
Show recommendations
 
 
 15. Skincare Recommendation Logic


Condition	Recommendation
Wrinkles	Retinol, hydration, sunscreen
Dark Spots	Vitamin C, Niacinamide
Puffy Eyes	Eye care, sleep
Clear Skin	Maintain routine
 
 
 16. Applications


Personal skincare monitoring
Beauty industry
Healthcare assistance
AI-based recommendation systems
  
  17. Advantages

Fast and automated
Easy to use
Cost-effective
Scalable
 
 
 18. Limitations


Depends on image quality
Limited categories
Not a medical replacement
 
 
 19. Future Enhancements


Mobile app
Cloud deployment
More skin conditions
Real-time webcam detection


20. Author


Vaishnavi Kesanakurthi

21.License

This project is based on Education purpose



22. Conclusion

DermalScan AI demonstrates how deep learning and computer vision can be applied to real-world skincare analysis, providing fast and accessible insights.
