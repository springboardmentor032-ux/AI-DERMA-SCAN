# Use TensorFlow image with CPU support
FROM tensorflow/tensorflow:2.12.0

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Streamlit default port
EXPOSE 8501

CMD ["streamlit", "run", "front_end.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
