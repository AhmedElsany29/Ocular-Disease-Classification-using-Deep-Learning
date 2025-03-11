import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
from flask import Flask, request, render_template_string
from PIL import Image
from io import BytesIO
import base64
import cv2

# Initialize Flask app
app = Flask(__name__)

# Load models
classification_model = load_model("model.h5")  # Path to the classification model
gradcam_model = EfficientNetB3(weights="imagenet")
last_conv_layer_name = "top_conv"

# Class mapping for classification model
class_mapping = {
    0: "ARMD",
    1: "Cataract",
    2: "Diabetic Retinopathy",
    3: "Glaucoma",
    4: "Normal",
}

# Grad-CAM function
def grad_cam(model, img_array, last_conv_layer_name):
    grad_model = Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = np.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.reduce_sum(heatmap, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

# Function to overlay Grad-CAM on the image
def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (300, 300))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    heatmap_img = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_img, alpha, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(superimposed_img)

# HTML Template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Classification & Grad-CAM Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background-color: #f4f4f9; margin: 0; padding: 20px; }
        h1 { color: #333; }
        form { margin: 20px auto; padding: 20px; background: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 5px; width: 50%; }
        input { margin: 10px 0; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .result { margin-top: 20px; font-size: 1.2em; color: #333; }
        img { margin-top: 20px; max-width: 400px; border: 2px solid #ddd; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Image Classification & Grad-CAM Visualization</h1>
    <form method="POST" enctype="multipart/form-data">
        <label for="image">Upload an Image:</label><br>
        <input type="file" name="image" required><br>
        <button type="submit">Submit</button>
    </form>

    {% if predicted_class %}
        <div class="result">
            <p><strong>Predicted Class:</strong> {{ predicted_class }}</p>
            <p><strong>Confidence:</strong> {{ confidence }}%</p>
            <h3>Uploaded Image:</h3>
            <img src="data:image/png;base64,{{ image_data }}" alt="Uploaded Image">
        </div>
    {% endif %}

    {% if gradcam_url %}
        <div class="result">
            <p>Grad-CAM Visualization:</p>
            <img src="{{ gradcam_url }}" alt="Grad-CAM">
        </div>
    {% endif %}
</body>
</html>
"""

# HTML Template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Classification & Grad-CAM Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background-color: #f4f4f9; margin: 0; padding: 20px; }
        h1 { color: #333; }
        form { margin: 20px auto; padding: 20px; background: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 5px; width: 50%; }
        input { margin: 10px 0; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .result { margin-top: 20px; font-size: 1.2em; color: #333; }
        .image-container { display: flex; justify-content: center; gap: 20px; }
        img { max-width: 800px; height: auto; border: 2px solid #ddd; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Image Classification & Grad-CAM Visualization</h1>
    <form method="POST" enctype="multipart/form-data">
        <label for="image">Upload an Image:</label><br>
        <input type="file" name="image" required><br>
        <button type="submit">Submit</button>
    </form>

    {% if predicted_class %}
        <div class="result">
            <p><strong>Predicted Class:</strong> {{ predicted_class }}</p>
            <p><strong>Confidence:</strong> {{ confidence }}%</p>
            <h3>Uploaded Image and Grad-CAM:</h3>
            <div class="image-container">
                <div>
                    <h4>Uploaded Image</h4>
                    <img src="data:image/png;base64,{{ image_data }}" alt="Uploaded Image">
                </div>
                <div>
                    <h4>Grad-CAM Visualization</h4>
                    <img src="{{ gradcam_url }}" alt="Grad-CAM">
                </div>
            </div>
        </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            return "No file uploaded", 400

        # Preprocess image for classification model (resize to 224x224)
        img = Image.open(file.stream).resize((224, 224))
        img_array = np.array(img)
        img_input_classification = np.expand_dims(img_array, axis=0)
        img_input_classification = preprocess_input(img_input_classification)

        # Convert image to Base64 to display in HTML
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Classification prediction
        predictions = classification_model.predict(img_input_classification)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_mapping[predicted_class_idx]
        confidence = round(float(predictions[0][predicted_class_idx]) * 100, 2)

        # Preprocess image for Grad-CAM model (resize to 300x300)
        img_resized_for_gradcam = img.resize((300, 300))  # Resize to 300x300 for Grad-CAM
        img_path = "uploaded_image.jpg"
        img_resized_for_gradcam.save(img_path)
        img_array_gradcam = np.expand_dims(np.array(img_resized_for_gradcam), axis=0)
        img_array_gradcam = preprocess_input(img_array_gradcam)

        # Generate Grad-CAM
        heatmap = grad_cam(gradcam_model, img_array_gradcam, last_conv_layer_name)
        gradcam_image = overlay_gradcam(img_path, heatmap)

        # Convert Grad-CAM image to Base64
        buffer = BytesIO()
        gradcam_image.save(buffer, format="PNG")
        gradcam_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

        return render_template_string(
            html_template,
            predicted_class=predicted_class,
            confidence=confidence,
            image_data=img_base64,
            gradcam_url=gradcam_url
        )

    return render_template_string(html_template)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
