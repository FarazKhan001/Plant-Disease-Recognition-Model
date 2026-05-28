from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import os
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploadimages')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

with open("plant_disease.json", 'r') as f:
    plant_disease = json.load(f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image_path):
    img = extract_features(image_path)
    prediction = model.predict(img)
    idx = int(prediction.argmax())
    if isinstance(plant_disease, list):
        return plant_disease[idx]
    return plant_disease.get(str(idx)) or plant_disease.get(idx)

def cleanup_old_uploads(keep_filename):
    """Delete all uploaded images except the one just saved."""
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname != keep_filename:
            try:
                os.remove(os.path.join(UPLOAD_FOLDER, fname))
            except OSError:
                pass

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method != "POST":
        return redirect('/')

    image = request.files.get('img')

    if not image or image.filename == '':
        return render_template('home.html', error="No file selected.")

    if not allowed_file(image.filename):
        return render_template('home.html', error="Only PNG and JPG files are allowed.")

    ext = image.filename.rsplit('.', 1)[1].lower()
    safe_filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    image.save(save_path)

    try:
        prediction = model_predict(save_path)
    except Exception as e:
        os.remove(save_path)
        return render_template('home.html', error=f"Prediction failed: {str(e)}")

    cleanup_old_uploads(safe_filename)

    image_url = f"/uploadimages/{safe_filename}"
    return render_template('home.html', result=True, imagepath=image_url, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
