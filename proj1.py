import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import tempfile

# Load YOLOv5 model (pre-trained on COCO dataset)
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

# Streamlit UI
st.title("🔍 Object Detection App")
st.write("Upload an image to detect objects using YOLOv5.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Confidence threshold slider
confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to numpy array
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    
    # Perform object detection
    results = model(img_array)
    
    # Filter results based on confidence threshold
    detections = results.pandas().xyxy[0]
    detections = detections[detections['confidence'] > confidence_threshold]
    
    # Draw bounding boxes
    for _, row in detections.iterrows():
        x1, y1, x2, y2, conf, cls = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"]), row["confidence"], row["name"]
        label = f"{cls} ({conf:.2f})"
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Convert back to PIL image
    detected_image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    
    # Show detected image
    st.image(detected_image, caption="Detected Objects", use_column_width=True)

    # Option to download the processed image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        detected_image.save(temp_file.name)
        st.download_button(label="📥 Download Processed Image", data=open(temp_file.name, "rb").read(), file_name="detected_image.jpg", mime="image/jpeg")
