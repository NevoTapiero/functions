import os
import tempfile
import tensorflow as tf
from firebase_admin import credentials, initialize_app, firestore, storage as firebase_storage
from flask import jsonify
from google.cloud import storage
from keras.models import load_model
from keras.preprocessing import image
from google.cloud.firestore import Increment
import datetime

# Initialize Firebase Admin with your Firebase config
cred = credentials.Certificate("msdk-app-3a2d5-fe27b66abfec.json")
app = initialize_app(cred, {
    'storageBucket': 'msdk-app-3a2d5.appspot.com'
})

# Define class labels
class_labels = [
    "Corn_common_rust",
    "Corn_healthy",
    "Corn_Infected",
    "Corn_northern_leaf_blight",
    "Corn_gray_leaf_spots"
]


def get_model():
    # Initialize the Google Cloud Storage client and get model from the bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket('msdk-app-3a2d5.appspot.com')
    model_blob = bucket.blob('mydataset_model.001.h5')
    with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
        model_blob.download_to_filename(temp_model_file.name)
        model = load_model(temp_model_file.name)
    os.remove(temp_model_file.name)  # Clean up the temporary model file
    return model


def process_images(user_id, batches):
    db = firestore.client(app=app)
    bucket = firebase_storage.bucket(app=app)

    model = get_model()

    # Reference to the document holding the counts
    count_ref = db.collection("Users").document(user_id).collection("count_classified_classes").document("countDict")

    for batch_name in batches:
        # Path in Firestore for the batch
        images_ref = db.collection("Users").document(user_id).collection("unclassified").document(
            batch_name).collection(
            "images")
        batch_ref = db.collection("Users").document(user_id).collection("unclassified").document(batch_name).get()
        batch_doc = batch_ref.to_dict()
        docs = images_ref.stream()

        for doc in docs:
            image_name = doc.id
            image_data = doc.to_dict()
            image_path = f"Users/{user_id}/unclassified/{image_name}"

            # Download image from Cloud Storage
            blob = bucket.blob(image_path)
            with tempfile.NamedTemporaryFile() as temp_file:
                blob.download_to_filename(temp_file.name)
                img = image.load_img(temp_file.name, target_size=(32, 32))
                img_array = image.img_to_array(img) / 255.0
                img_array = tf.expand_dims(img_array, 0)  # Model expects a batch

                predictions = model.predict(img_array)
                predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
                predicted_class = class_labels[predicted_class_index]

                # Update the document data with the new classTag
                image_data['classTag'] = predicted_class

                # Prepare the update dictionary using Increment for atomic increments
                update_dict = {predicted_class: Increment(1)}

                # Atomically update the count for the predicted class
                count_ref.set(update_dict, merge=True)

                new_image_ref = db.collection("Users").document(user_id).collection(predicted_class).document(
                    batch_name).collection("images").document(image_name)

                new_image_ref.set(image_data)

                new_batch_ref = db.collection("Users").document(user_id).collection(predicted_class).document(
                    batch_name)
                new_batch_ref.set(batch_doc)

                # Delete the original document
                doc.reference.delete()

                # Move the image in Cloud Storage
                new_image_path = f"Users/{user_id}/{predicted_class}/{image_name}"

                new_blob = bucket.blob(new_image_path)

                new_blob.rewrite(blob)

                # Get the new download URL
                new_url = new_blob.generate_signed_url(version="v4", expiration=datetime.timedelta(minutes=10),
                                                       method='GET')

                # Update Firestore document with the new URL
                new_image_ref.update({'imageUrl': new_url})

                blob.delete()
    for batch_name in batches:
        db.collection("Users").document(user_id).collection("unclassified").document(batch_name).delete()


def classify_and_move_all_unclassified_images(request):
    if request.method != 'POST':
        return 'Only POST requests are accepted', 405

    user_id = request.json.get('user_id')
    batches = request.json.get('batches')

    process_images(user_id, batches)

    return jsonify({"message": "Processed all unclassified images"})
