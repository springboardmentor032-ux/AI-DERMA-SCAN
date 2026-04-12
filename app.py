from flask import Flask, render_template, request, url_for
import os
import cv2
import numpy as np
import csv
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('skin_aging_model.h5')
categories = ['Clear Skin', 'Dark Spots', 'Puffy Eyes', 'Wrinkles']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/', methods=['GET', 'POST'])
def index():
    results_list = []
    filename = None
    
    if request.method == 'POST':
        file = request.files['file']
        if file:
            upload_dir = 'static/uploads'
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            
            filepath = os.path.join(upload_dir, file.filename)
            file.save(filepath)
            filename = file.filename
            
            image = cv2.imread(filepath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = image[y:y+h, x:x+w]
                img_ready = cv2.resize(face_roi, (224, 224))
                img_ready = img_ready.astype('float32') / 255.0
                img_ready = np.expand_dims(img_ready, axis=0)
                
                preds = model.predict(img_ready)[0]
                
                
                for i in range(len(categories)):
                    results_list.append({
                        'category': categories[i],
                        'percentage': round(float(preds[i]) * 100, 1)
                    })
                
                results_list = sorted(results_list, key=lambda x: x['percentage'], reverse=True)
                
                
                with open('static/report.csv', mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([filename, results_list[0]['category'], results_list[0]['percentage']])

    return render_template('index.html', filename=filename, results=results_list)

if __name__ == '__main__':
    app.run(debug=True)