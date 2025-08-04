from flask import Flask, request, jsonify
import cv2
import firebase_admin
from firebase_admin import credentials, firestore
from imgbeddings import imgbeddings
from PIL import Image
import os
import io

app = Flask(__name__)

# --- FIREBASE INITIALIZATION ---
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Embedding Model ---
ibed = imgbeddings()

@app.route('/upload', methods=['POST'])
def upload_face():
    if 'image' not in request.files or 'name' not in request.form or 'details' not in request.form:
        return jsonify({"error": "Missing required parameters."}), 400

    file = request.files['image']
    name = request.form['name']
    details = request.form['details']

    try:
        image = Image.open(file.stream).convert('RGB')
        embedding = ibed.to_embeddings(image)[0].tolist()

        doc_ref = db.collection('persons_mediapipe').document(name.replace(" ", "_"))
        doc_ref.set({
            'name': name,
            'details': details,
            'face_embedding': embedding,
        })
        return jsonify({"message": f"Successfully added {name}."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "Face Embedding API is running.", 200

if __name__ == '__main__':
    app.run(debug=True)
