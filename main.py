import os
import tempfile
import tensorflow as tf
from firebase_admin import credentials, firestore, initialize_app
from flask import jsonify
from google.cloud import storage, firestore
from keras.models import load_model
from keras.preprocessing import image

# Initialize Firebase Admin with your Firebase config
cred = credentials.Certificate("msdk-app-3a2d5-fe27b66abfec.json")

initialize_app(cred)

# Define class labels (the same labels used during training)
class_labels = [
    "Corn_common_rust",
    "Corn_healthy",
    "Corn_Infected",
    "Corn_northern_leaf_blight",
    "Corn_gray_leaf_spots"
]


def get_model(bucket):
    # Assumes model is stored in Cloud Storage
    model_blob = bucket.blob('mydataset_model.001.h5')
    with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
        model_blob.download_to_filename(temp_model_file.name)
        model = load_model(temp_model_file.name)
    os.remove(temp_model_file.name)  # Clean up the temporary model file
    return model


def classify_and_move_all_unclassified_images(request):
    if request.method != 'POST':
        return 'Only POST requests are accepted', 405

    # Initialize the Google Cloud Storage client
    storage_client = storage.Client()
    bucket_name = 'msdk-app-3a2d5.appspot.com'
    bucket = storage_client.bucket(bucket_name)

    # Initialize Firestore client
    db = firestore.Client()

    # Load the model once at the beginning
    model = get_model(bucket)

    # List and process each image blob directly under 'unclassified/'
    blobs = bucket.list_blobs(prefix="unclassified/")
    for blob in blobs:
        file_path = blob.name
        print("the blob:" + "".join(blob.name))
        if not file_path.lower().endswith(('.jpg', '.png')):
            try:
                # Download, preprocess, and predict class for the image
                with tempfile.NamedTemporaryFile(suffix=os.path.basename(file_path)) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    img = image.load_img(temp_file.name, target_size=(32, 32))
                    img_array = image.img_to_array(img) / 255.0
                    img_array = tf.expand_dims(img_array, 0)  # Model expects a batch

                    predictions = model.predict(img_array)

                    # Predicted class from the model
                    predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
                    predicted_class = class_labels[predicted_class_index]

                    parts = file_path.split('/')
                    imageName = parts[-1]  # This gets the last part of the path, which should be the image name

                    # Constructing the new path for the blob based on its predicted class
                    new_blob_path = f"classified/{predicted_class}/{imageName}"

                    # Copy the blob to the new location in the 'classified' directory under the specific class
                    new_blob = bucket.copy_blob(blob, bucket, new_blob_path)

                    # Delete the original blob from the 'unclassified' directory
                    blob.delete()

                    # Query Firestore for all documents with the imageName across all batches
                    docs = db.collection_group('images').where('imageName', '==', imageName).stream()

                    for doc in docs:
                        doc_ref = doc.reference  # Get the document reference
                        doc_data = doc.to_dict()

                        # Update the document data with the new classTag
                        doc_data['classTag'] = predicted_class

                        doc_ref_classified = db.collection(predicted_class).document(imageName)

                        # Set the document in the new location
                        doc_ref_classified.set(doc_data)

                        # Delete the original document
                        doc_ref.delete()

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return jsonify({"message": "Processed all unclassified images"})
