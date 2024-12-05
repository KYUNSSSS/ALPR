import cv2
import pathlib
import torch
import re
import os
import streamlit as st
from paddleocr import PaddleOCR

# Set environment variables to avoid OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Fix the PosixPath error on Windows systems
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the YOLOv5 model
@st.cache_resource
def load_yolo_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

model = load_yolo_model()

# Initialize PaddleOCR with English model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def filter_text(text):
    """Filters the text to only include patterns with an alphabet followed by digits."""
    pattern = r'[A-Z0-9]'
    matches = re.findall(pattern, text)
    return ''.join(matches)

def detect_plate_and_recognize_text(image_path):
    """Detects license plates in an image and recognizes the text."""
    img = cv2.imread(image_path)
    results = model(img)
    detected_texts = []
    plate_detected = False

    for box in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if confidence < 0.1:
            continue
        plate_detected = True

        # Crop the detected plate
        plate_image = img[y1:y2, x1:x2]

        # Use PaddleOCR to detect and recognize text
        ocr_results = ocr.ocr(plate_image, cls=True)

        if ocr_results and isinstance(ocr_results, list):
            for result in ocr_results:
                if result and isinstance(result, list):
                    for line in result:
                        if line and len(line) > 1:
                            text = line[1][0]
                            filtered_text = filter_text(text)
                            if filtered_text:  # Only append if filtered text is not empty
                                detected_texts.append(filtered_text)

                                # Draw bounding box and text on the image (optional)
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(img, filtered_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                            (255, 0, 0), 2)

    # Save the processed image
    output_path = 'detected_plate_image.jpg'
    cv2.imwrite(output_path, img)

    if not plate_detected:
        detected_texts.append("Plate not Detected")

    if not detected_texts:
        detected_texts.append("Text not Detected")

    return output_path, detected_texts


# Streamlit App
st.title("License Plate Detection and Recognition")
st.write("Upload an image to detect license plates and recognize the text.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image("uploaded_image.jpg", caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Perform detection and recognition
    detected_image_path, detected_texts = detect_plate_and_recognize_text("uploaded_image.jpg")

    # Display results
    st.image(detected_image_path, caption="Processed Image with Detected Plates", use_column_width=True)
    st.write("**Detected Texts:**")
    for text in detected_texts:
        st.write(f"- {text}")
