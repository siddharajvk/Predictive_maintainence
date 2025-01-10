import base64
import io
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure TensorFlow to handle the deprecated keywords
def load_keras_model():
    try:
        # First attempt: Load with custom object scope
        return load_model('model.h5', compile=False)
    except Exception as e:
        print(f"First attempt failed: {str(e)}")
        try:
            # Second attempt: Load with custom configuration
            return tf.keras.models.load_model(
                'model.h5',
                custom_objects=None,
                compile=False
            )
        except Exception as e:
            print(f"Second attempt failed: {str(e)}")
            # Third attempt: Create and load weights directly
            try:
                # Define the model architecture manually
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),  # Adjust input_shape based on your model
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])
                
                # Load weights
                model.load_weights('model.h5')
                return model
            except Exception as e:
                print(f"Third attempt failed: {str(e)}")
                raise Exception("Could not load model using any method")

# Load models
try:
    yolo_model = YOLO('best.pt')
    keras_model = load_keras_model()
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

# Get the class names from YOLO model for counting
class_list = yolo_model.names

def count_objects_per_class(detections):
    """
    Count the number of detected objects per class
    """
    class_counts = {class_name: 0 for class_name in class_list.values()}
    for detection in detections:
        class_name = detection['class']
        class_counts[class_name] += 1
    return class_counts

def prepare_ml_input(class_counts, feature1, feature2):
    """
    Prepare the input for the ML model in the specified order:
    feature1, feature2, Sidewall_cracks, Puncture, Hairline
    """
    # Get class counts for each class, defaulting to 0 if not present
    sidewall_cracks_count =float(class_counts['Sidewall Crack']*0.7)
    puncture_count = float(class_counts['Puncture'])
    hairline_count = float(class_counts['Hairline Crack']*0.3)
    print(sidewall_cracks_count,puncture_count,hairline_count)
    # Combine feature1, feature2, and class counts (no weighting applied)
    input_array = np.array([feature1, feature2, sidewall_cracks_count, puncture_count,hairline_count], dtype=float)

    return np.expand_dims(input_array, axis=0)

def normalize_input(input_array):
    """
    Normalize the input data
    """
    # Simple min-max normalization, adjust based on your training data
    return input_array / np.max(input_array) if np.max(input_array) > 0 else input_array

@app.route('/')
def index():
    return render_template('index.html')

scaler = StandardScaler()

def scale_input(input_array):
    """
    Scale the input data using StandardScaler and round to 8 decimal places.
    """
    # Reshape to fit StandardScaler requirements
    # reshaped_array = input_array.reshape(-1, input_array.shape[-1])
    data = pd.read_csv('tyre_data_diverse_target2.csv')
    features = ['Age', 'KM', 'Puncture(Weight)', 'Sidewall(Weight)', 'Hairline(Weight)']
    X = data[features]
    y = data['Years_Replacement']

    # Fit and transform the input using StandardScaler
    X_Scaled = scaler.fit_transform(X)
    scaled_array=scaler.transform(input_array)
    print("asdasd",input_array)
    
    # Round to 8 decimal places
    # scaled_array = np.round(scaled_array, decimals=8)
    print("Scaled array is",scaled_array)
    return scaled_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image and additional features from POST request
        file = request.files['image']
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        if not file:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            })

        # Save the file temporarily
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.jpg')
        file.save(img_path)
        
        # Make YOLO prediction
        results = yolo_model(img_path)
        result = results[0]
        
        # Get boxes, confidence scores and class names
        boxes = result.boxes.xyxy.tolist()  # Get box coordinates
        confidences = result.boxes.conf.tolist()  # Get confidence scores
        class_names = [result.names[int(c)] for c in result.boxes.cls.tolist()]  # Get class names
        
        # Create detections list
        detections = []
        for box, conf, class_name in zip(boxes, confidences, class_names):
            detections.append({
                'box': box,
                'confidence': float(conf),
                'class': class_name
            })
        
        # Count objects per class
        class_counts = count_objects_per_class(detections)
        
        # Print debug information
        print("Class counts:", class_counts)
        
        # Prepare input for ML model
        print(class_counts)
        ml_input = prepare_ml_input(class_counts, feature1, feature2)
        print("ML input shape:", ml_input.shape)
        print("ML input values:", ml_input)
        
        # Scale input
        ml_input_scaled = scale_input(ml_input)
        print("Scaled input:", ml_input_scaled)
        
        # Make prediction
        years_replacement = float(keras_model.predict(ml_input_scaled, verbose=0)[0][0])
        print("Prediction:", years_replacement)
        
        # Read the image and convert to base64
        with open(img_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        return jsonify({
            'success': True,
            'detections': detections,
            'class_counts': class_counts,
            'years_replacement': years_replacement,
            'image': img_data
        })

    except Exception as e:
        import traceback
        print("Error occurred:", str(e))
        print("Traceback:", traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        })
if __name__ == '__main__':
    app.run(debug=True)
