from flask import Flask, render_template, request
import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TENSORFLOW_AVAILABLE = True
except ModuleNotFoundError:
    TENSORFLOW_AVAILABLE = False

app = Flask(__name__)

# TensorFlow model and preprocessing setup
if TENSORFLOW_AVAILABLE:
    model_path = "fine_tuned_flood_detection_model.h5"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        labels = ['Flooding', 'No Flooding']

        def preprocess_image(image_path):
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    else:
        model = None
        labels = []
        TENSORFLOW_AVAILABLE = False
else:
    model = None
    labels = []

def error_message(msg):
    return render_template('prediction_result.html', result={
        'label': 'Error',
        'confidence': msg,
        'image_path': ''
    })

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('image_upload.html')

    if not TENSORFLOW_AVAILABLE or model is None:
        return error_message("TensorFlow or model file not available in this environment.")

    if 'image' not in request.files:
        return error_message("No file uploaded")

    image = request.files['image']
    if image.filename == '':
        return error_message("Empty filename")

    image_path = os.path.join('static', image.filename)
    image.save(image_path)

    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index] * 100

    result = {
        'label': labels[predicted_index],
        'confidence': f"{confidence:.2f}%",
        'image_path': image_path
    }
    return render_template('prediction_result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
