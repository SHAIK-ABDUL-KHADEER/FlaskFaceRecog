import os
import cv2
import face_recognition
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Define the path to your dataset directory
dataset_path = r"C:\Users\sak78\PycharmProjects\pythonProject\dataset"

# Load the dataset
def load_dataset(dataset_path):
    dataset = {}
    for person_folder in os.listdir(dataset_path):
        person_name = person_folder
        person_images = []

        for image_file in os.listdir(os.path.join(dataset_path, person_folder)):
            image_path = os.path.join(dataset_path, person_folder, image_file)

            # Load the image
            img = face_recognition.load_image_file(image_path)
            person_images.append(img)

        dataset[person_name] = person_images

    return dataset

# Load the dataset
dataset = load_dataset(dataset_path)

# Function to perform face recognition using dlib's deep learning model
def recognize_face(input_image):
    # Find face locations in the input image
    face_locations = face_recognition.face_locations(input_image)

    if not face_locations:
        return "Not authorized", None

    # Encode the faces in the input image
    input_face_encodings = face_recognition.face_encodings(input_image, face_locations)

    # Compare the input face encodings with known face encodings
    # Compare the input face encodings with known face encodings
    for dataset_person_name, person_images in dataset.items():
        for person_image in person_images:
            # Encode the known faces
            known_face_encodings = face_recognition.face_encodings(person_image)

            # Compare the input face encodings with known face encodings
            for known_face_encoding in known_face_encodings:
                matches = face_recognition.compare_faces([known_face_encoding], input_face_encodings[0],
                                                         tolerance=0.4)  # Adjust the tolerance value
                if any(matches):
                    return "Authorized", dataset_person_name  # Use the dataset person name here

    return "Not authorized", None


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded image file
        file = request.files["image"]

        if file:
            # Read the uploaded image
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            rgb_input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform face recognition
            status, matched_person = recognize_face(rgb_input_image)

            if status == "Authorized":
                result = f"Authorized: {matched_person}"
            else:
                result = "Not authorized"

            return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
