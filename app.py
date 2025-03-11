import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask import Flask, request, render_template_string
from PIL import Image
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = load_model("model.h5") 

# Class mapping
class_mapping = {
    0: "ARMD",
    1: "Cataract",
    2: "Diabetic Retinopathy",
    3: "Glaucoma",
    4: "Normal",
}

# HTML Template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Classification</title>
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
    <h1>Image Classification</h1>
    <form method="POST" enctype="multipart/form-data">
        <label for="image">Upload an Image:</label><br>
        <input type="file" name="image" required><br>
        <button type="submit">Classify</button>
    </form>
    {% if predicted_class %}
        <div class="result">
            <p><strong>Predicted Class:</strong> {{ predicted_class }}</p>
            <p><strong>Confidence:</strong> {{ confidence }}%</p>
            <h3>Uploaded Image:</h3>
            <img src="data:image/png;base64,{{ image_data }}" alt="Uploaded Image">
        </div>
    {% endif %}
</body>
</html>
"""

# Flask routes
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get uploaded image
        file = request.files.get("image")
        if not file:
            return "No image uploaded", 400

        # Preprocess the image
        img = Image.open(file.stream).resize((224, 224))
        img_array = np.array(img)
        img_input = np.expand_dims(img_array, axis=0)
        img_input = preprocess_input(img_input)

        # Convert image to Base64 to display in HTML
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Make prediction
        predictions = model.predict(img_input)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_mapping[predicted_class_idx]
        confidence = round(float(predictions[0][predicted_class_idx]) * 100, 2)

        return render_template_string(
            html_template,
            predicted_class=predicted_class,
            confidence=confidence,
            image_data=img_base64  # Pass Base64 image to the template
        )

    return render_template_string(html_template)

if __name__ == "__main__":
    app.run(debug=True, port=8088)

