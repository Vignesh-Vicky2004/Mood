from flask import Flask, request, jsonify
import os
import uuid
import firebase_admin
from firebase_admin import credentials, firestore, storage
from imgbeddings import imgbeddings
from PIL import Image
import io
import base64
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# --- CONFIGURATION ---
SERVICE_ACCOUNT_KEY_PATH = 'serviceAccountKey.json'
STORAGE_BUCKET_NAME = 'gs://moodmate-a5ca1.firebasestorage.app'  # Replace with your actual bucket name

# --- FIREBASE INITIALIZATION ---
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred, {
            'storageBucket': STORAGE_BUCKET_NAME
        })
        logger.info("✅ Firebase App initialized successfully.")
except Exception as e:
    logger.error(f"❌ Error initializing Firebase App: {e}")
    exit(1)

# Initialize Firebase services
db = firestore.client()
bucket = storage.bucket()
ibed = imgbeddings()


# --- UTILITY FUNCTIONS ---
def upload_image_to_storage(image_file, folder_name="face_images"):
    """
    Upload image to Firebase Storage and return the download URL
    """
    try:
        # Generate unique filename
        file_extension = image_file.filename.split('.')[-1] if '.' in image_file.filename else 'jpg'
        unique_filename = f"{folder_name}/{uuid.uuid4()}.{file_extension}"

        # Upload to Firebase Storage
        blob = bucket.blob(unique_filename)
        blob.upload_from_file(image_file, content_type=f'image/{file_extension}')

        # Make the blob publicly readable
        blob.make_public()

        return blob.public_url
    except Exception as e:
        logger.error(f"Error uploading image to storage: {e}")
        raise


def process_image_embedding(image_file):
    """
    Generate embedding from uploaded image
    """
    try:
        # Reset file pointer
        image_file.seek(0)

        # Open image with PIL
        img = Image.open(image_file)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Generate embedding
        embedding = ibed.to_embeddings(img)[0].tolist()

        return embedding
    except Exception as e:
        logger.error(f"Error processing image embedding: {e}")
        raise


def validate_image_file(file):
    """
    Validate uploaded image file
    """
    if not file:
        return False, "No file provided"

    if file.filename == '':
        return False, "No file selected"

    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''

    if file_extension not in allowed_extensions:
        return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"

    # Check file size (max 10MB)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size > 10 * 1024 * 1024:  # 10MB
        return False, "File size exceeds 10MB limit"

    return True, "Valid file"


# --- API ENDPOINTS ---

@app.route('/health', methods=['GET'])
def health_check():
    """
    Basic health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Firebase Face Recognition API",
        "version": "1.0.0"
    }), 200


@app.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    """
    Detailed health check with Firebase connectivity
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Firebase Face Recognition API",
        "version": "1.0.0",
        "checks": {}
    }

    # Check Firestore connectivity
    try:
        # Try to read from a test collection
        test_ref = db.collection('health_check').limit(1)
        list(test_ref.stream())
        health_status["checks"]["firestore"] = "connected"
    except Exception as e:
        health_status["checks"]["firestore"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    # Check Firebase Storage connectivity
    try:
        # Try to list blobs (this checks connectivity)
        blobs = list(bucket.list_blobs(max_results=1))
        health_status["checks"]["storage"] = "connected"
    except Exception as e:
        health_status["checks"]["storage"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    # Check embedding model
    try:
        # Create a small test image to verify the model works
        test_img = Image.new('RGB', (100, 100), color='red')
        test_embedding = ibed.to_embeddings(test_img)
        health_status["checks"]["embedding_model"] = "loaded"
    except Exception as e:
        health_status["checks"]["embedding_model"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code


@app.route('/enroll', methods=['POST'])
def enroll_person():
    """
    Enroll a new person with their face image
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        name = request.form.get('name', '').strip()
        details = request.form.get('details', '').strip()

        # Validate inputs
        if not name:
            return jsonify({"error": "Name is required"}), 400

        # Validate image file
        is_valid, validation_message = validate_image_file(image_file)
        if not is_valid:
            return jsonify({"error": validation_message}), 400

        # Check if person already exists
        doc_id = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        existing_doc = db.collection('persons_mediapipe').document(doc_id).get()
        if existing_doc.exists:
            return jsonify({"error": f"Person '{name}' already exists"}), 409

        # Process image embedding
        embedding = process_image_embedding(image_file)

        # Upload image to Firebase Storage
        image_file.seek(0)  # Reset file pointer
        image_url = upload_image_to_storage(image_file, "enrolled_faces")

        # Save to Firestore
        person_data = {
            'name': name,
            'details': details,
            'face_embedding': embedding,
            'image_url': image_url,
            'enrolled_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP
        }

        db.collection('persons_mediapipe').document(doc_id).set(person_data)

        logger.info(f"Successfully enrolled person: {name}")

        return jsonify({
            "message": f"Successfully enrolled '{name}'",
            "person_id": doc_id,
            "image_url": image_url,
            "embedding_size": len(embedding)
        }), 201

    except Exception as e:
        logger.error(f"Error enrolling person: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/recognize', methods=['POST'])
def recognize_person():
    """
    Recognize a person from an uploaded image
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        threshold = float(request.form.get('threshold', 0.8))  # Similarity threshold

        # Validate image file
        is_valid, validation_message = validate_image_file(image_file)
        if not is_valid:
            return jsonify({"error": validation_message}), 400

        # Process image embedding
        query_embedding = process_image_embedding(image_file)

        # Upload query image to storage for logging
        image_file.seek(0)
        query_image_url = upload_image_to_storage(image_file, "query_images")

        # Get all enrolled persons
        persons_ref = db.collection('persons_mediapipe')
        enrolled_persons = persons_ref.stream()

        best_match = None
        best_similarity = 0.0

        # Compare with all enrolled persons
        for person_doc in enrolled_persons:
            person_data = person_doc.to_dict()
            stored_embedding = person_data.get('face_embedding', [])

            if not stored_embedding:
                continue

            # Calculate cosine similarity
            similarity = calculate_cosine_similarity(query_embedding, stored_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    'person_id': person_doc.id,
                    'name': person_data.get('name'),
                    'details': person_data.get('details'),
                    'similarity': similarity,
                    'enrolled_image_url': person_data.get('image_url')
                }

        # Log recognition attempt
        recognition_log = {
            'query_image_url': query_image_url,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'best_match': best_match,
            'threshold_used': threshold,
            'recognition_successful': best_match and best_similarity >= threshold
        }

        db.collection('recognition_logs').add(recognition_log)

        if best_match and best_similarity >= threshold:
            return jsonify({
                "recognized": True,
                "person": best_match,
                "query_image_url": query_image_url
            }), 200
        else:
            return jsonify({
                "recognized": False,
                "best_match": best_match,
                "message": "No matching person found above threshold",
                "query_image_url": query_image_url
            }), 200

    except Exception as e:
        logger.error(f"Error recognizing person: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/persons', methods=['GET'])
def list_persons():
    """
    List all enrolled persons
    """
    try:
        persons_ref = db.collection('persons_mediapipe')
        enrolled_persons = persons_ref.stream()

        persons_list = []
        for person_doc in enrolled_persons:
            person_data = person_doc.to_dict()
            # Remove embedding from response (too large)
            person_data.pop('face_embedding', None)
            person_data['person_id'] = person_doc.id
            persons_list.append(person_data)

        return jsonify({
            "persons": persons_list,
            "total_count": len(persons_list)
        }), 200

    except Exception as e:
        logger.error(f"Error listing persons: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/persons/<person_id>', methods=['GET'])
def get_person(person_id):
    """
    Get details of a specific person
    """
    try:
        person_doc = db.collection('persons_mediapipe').document(person_id).get()

        if not person_doc.exists:
            return jsonify({"error": "Person not found"}), 404

        person_data = person_doc.to_dict()
        # Remove embedding from response (too large)
        person_data.pop('face_embedding', None)
        person_data['person_id'] = person_id

        return jsonify({"person": person_data}), 200

    except Exception as e:
        logger.error(f"Error getting person: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/persons/<person_id>', methods=['DELETE'])
def delete_person(person_id):
    """
    Delete a person from the database
    """
    try:
        person_doc = db.collection('persons_mediapipe').document(person_id).get()

        if not person_doc.exists:
            return jsonify({"error": "Person not found"}), 404

        person_data = person_doc.to_dict()
        person_name = person_data.get('name', 'Unknown')

        # Delete the document
        db.collection('persons_mediapipe').document(person_id).delete()

        logger.info(f"Successfully deleted person: {person_name} (ID: {person_id})")

        return jsonify({
            "message": f"Successfully deleted person '{person_name}'",
            "person_id": person_id
        }), 200

    except Exception as e:
        logger.error(f"Error deleting person: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


def calculate_cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings
    """
    import numpy as np

    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # For development only
    app.run(debug=True, host='0.0.0.0', port=5000)