from flask import Flask, render_template, request, jsonify
from chat import get_response
import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the dental caries detection model
model = load_model('dental_caries_detection_model.h5')

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict_text():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

@app.route("/predict_image", methods=["POST"])
def predict_image():
    file = request.files["file"]

    upload_folder = "uploads"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)

    if prediction < 0.5:
        result = "Our analysis indicates the presence of dental caries. We recommend scheduling an appointment with a dental professional for further evaluation and treatment."
    else:
        result = "No signs of dental caries detected. Remember to continue practicing good oral hygiene habits for optimal dental health."

    # Delete the uploaded image file
    os.remove(file_path)

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=False)
