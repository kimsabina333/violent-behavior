import os
import io
from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image, ImageDraw
import torch
import numpy as np
from ultralytics import YOLO
from flask_pymongo import PyMongo
from datetime import datetime

# Initialize the Flask application
app = Flask(__name__)

# Configure MongoDB URI
app.config["MONGO_URI"] = "mongodb://localhost:27017/yourdatabase"
mongo = PyMongo(app)

# Load the YOLOv8 model
model = YOLO('best.pt')

# Define the preprocess_image function
def preprocess_image(image):
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image

# Define the draw_boxes function
def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    for detection in detections:
        bbox = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']
        draw.rectangle(bbox, outline='red', width=3)
        draw.text((bbox[0], bbox[1]), f"{label} ({confidence:.2f})", fill='red')
    return image

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file provided'}), 400

    image = Image.open(io.BytesIO(file.read()))
    input_image = preprocess_image(image)

    results = model(input_image)

    detections = []
    for result in results:
        for box in result.boxes:
            label = int(box.cls)
            confidence = float(box.conf)
            bbox = box.xyxy[0].cpu().numpy().tolist()
            detections.append({'label': model.names[label], 'confidence': confidence, 'bbox': bbox})

    output_image = draw_boxes(image, detections)

    # Save output image to "uploads" directory
    output_path = os.path.join("uploads", f"output_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
    output_image.save(output_path)

    # Save image and prediction to the database
    user_data = {
        "Username": request.form.get('username'),
        "Email": request.form.get('email')
    }
    user = mongo.db.users.find_one({"Email": user_data['Email']})
    if not user:
        user_id = mongo.db.users.insert_one(user_data).inserted_id
    else:
        user_id = user['_id']

    image_data = {
        "User_ID": user_id,
        "Image_URL": output_path,  # Save path to output image
        "Uploaded_At": datetime.utcnow(),
        "Processed_At": None
    }
    image_id = mongo.db.images.insert_one(image_data).inserted_id

    # Extract the prediction result and set the category ID accordingly
    if detections:
        prediction_result = detections[0]['label']  # Assuming the first detection is the primary prediction
        confidence_level = max(d['confidence'] for d in detections)
        # Set category ID based on prediction result
        category_id = get_category_id(prediction_result)
    else:
        prediction_result = "No detection"
        confidence_level = 0.0
        category_id = 4  # Assuming "Empty" category for no detection

    prediction_data = {
        "Image_ID": image_id,
        "Prediction_Result": prediction_result,
        "Confidence_Level": confidence_level,
        "Predicted_At": datetime.utcnow(),
        "Category_ID": category_id  # Set category ID
    }
    mongo.db.predictions.insert_one(prediction_data)

    return send_file(output_path, mimetype='image/jpeg')

# Function to get category ID based on prediction result
def get_category_id(prediction_result):
    if prediction_result == "knife":
        return 1
    elif prediction_result == "Violence":
        return 2
    elif prediction_result == "NonViolence":
        return 3
    elif prediction_result == "Empty":
        return 4
    elif prediction_result == "gun":
        return 5
    else:
        return 4  # Default to "Empty" category if not recognized

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
