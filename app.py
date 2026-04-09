import streamlit as st
import numpy as np
import cv2
import pandas as pd
from utils import predict_image

st.set_page_config(page_title="DermalScan AI", layout="centered")

# -----------------------------
# MODERN DARK BACKGROUND
# -----------------------------
st.markdown("""
<style>

/* Animated background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #020617);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Container */
[data-testid="block-container"] {
    max-width: 700px;
}

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.4);
    margin-top: 20px;
    animation: fadeIn 1s ease;
}

@keyframes fadeIn {
    from {opacity:0; transform: translateY(20px);}
    to {opacity:1; transform: translateY(0);}
}

/* Title */
.title {
    text-align:center;
    font-size:38px;
    font-weight:700;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Text */
p, h1, h2, h3 {
    color: white;
}

/* Upload box */
.stFileUploader {
    border: 2px dashed #38bdf8;
    border-radius: 12px;
    padding: 15px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 10px;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER (ALWAYS VISIBLE)
# -----------------------------
st.markdown("""
<div class="card">
    <div class="title">✨ DermalScan AI</div>
    <p style="text-align:center; color:#cbd5f5;">
    AI-powered facial skin analysis
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# UPLOAD ONLY (INITIAL VIEW)
# -----------------------------
st.markdown('<div class="card"><h3>📤 Upload Face Image</h3></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

# -----------------------------
# STOP HERE UNTIL USER UPLOADS
# -----------------------------
if uploaded_file is None:
    st.stop()

# -----------------------------
# PROCESS IMAGE AFTER UPLOAD
# -----------------------------
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
image = cv2.imdecode(file_bytes, 1)

st.image(image, caption="Uploaded Image", use_container_width=True)

with st.spinner("🔍 Analyzing..."):
    results, probs = predict_image(image)

if len(results) == 0:
    st.error("❌ No face detected")
    st.stop()

labels = []

for (x, y, w, h, label) in results:
    labels.append(label)

    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(image, label, (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

st.image(image, channels="BGR", caption="Detection Result")

final_label = labels[0].split(" ")[0]

# -----------------------------
# REPORT SECTION (SHOW AFTER UPLOAD)
# -----------------------------
st.markdown('<div class="card"><h2>🧑‍⚕️ Skin Report</h2></div>', unsafe_allow_html=True)
st.success(labels[0])

# -----------------------------
# CHART
# -----------------------------
classes = ["clear_skin", "dark_spots", "puffy_eyes", "wrinkles"]

df = pd.DataFrame({
    "Condition": classes,
    "Confidence": probs
})

st.markdown("### 📊 Confidence Analysis")
st.bar_chart(df.set_index("Condition"))

# -----------------------------
# SOLUTIONS
# -----------------------------
def get_skin_solution(label):
    return {
        "wrinkles": ("Aging signs detected", ["Use retinol", "Apply sunscreen", "Hydrate"]),
        "dark_spots": ("Pigmentation detected", ["Vitamin C", "Niacinamide", "SPF"]),
        "puffy_eyes": ("Under-eye swelling", ["Eye cream", "Sleep well", "Cold compress"]),
        "clear_skin": ("Healthy skin", ["Maintain routine", "Hydration", "SPF"])
    }.get(label, ("", []))

problem, routine = get_skin_solution(final_label)

st.markdown("### 💡 Skincare Tips")
st.write(problem)

for tip in routine:
    st.write("✔️", tip)