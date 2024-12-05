
from flask import Flask, render_template, request, redirect, url_for
import cv2
import pathlib
import torch
import re
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from paddleocr import PaddleOCR

# Set the environment variable to avoid OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Create the folder if it doesn't exist

# Fix the PosixPath error on Windows systems
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

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

        # Check if ocr_results is valid and not empty
        if ocr_results and isinstance(ocr_results, list):
            for result in ocr_results:
                # Check if result is valid and iterable
                if result and isinstance(result, list):
                    for line in result:
                        if line and len(line) > 1:
                            text = line[1][0]
                            filtered_text = filter_text(text)
                            if filtered_text:  # Only append if filtered text is not empty
                                detected_texts.append(filtered_text)

                            # Draw bounding box and text on the image (optional)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, filtered_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Save the processed image with bounding boxes and recognized text
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_plate_image.jpg')
    cv2.imwrite(output_path, img)
    if not plate_detected:
        detected_texts.append("Plate not Detected")
    # Check if no text was detected
    if not detected_texts:
        detected_texts.append("Text not Detected")

    return 'detected_plate_image.jpg', detected_texts

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            uploaded_image, detected_texts = detect_plate_and_recognize_text(file_path)
            return render_template('index.html', uploaded_image=uploaded_image, texts=detected_texts)

    # When GET request or no file uploaded, render the template without displaying an image
    return render_template('index.html', uploaded_image=None, texts=[])

if __name__ == '__main__':
    app.run(debug=True)
