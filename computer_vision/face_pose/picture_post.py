import face_recognition
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Initialization
print("Initializing...")
model = YOLO('yolov8n-pose.pt')

# Load known faces
print("Loading training images...")
known_face_encodings = []
known_face_names = []
if os.path.exists('training/'):
    for filename in os.listdir('training/'):
        if filename.endswith('.jpg'):
            print(f"Processing {filename}...")
            image_path = os.path.join('training/', filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                face_encoding = encodings[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append("Clayton Thompson")
            else:
                print(f"No faces found in {filename}")
print(f"Loaded {len(known_face_encodings)} face encodings.")

# Load a test image
test_image_path = '7cakQHm7_400x400.jpg'
test_image = cv2.imread(test_image_path)

# Run YOLO pose estimation
print("Running YOLO pose estimation...")
yolo_results = model.predict(test_image)
yolo_annotated_frame = yolo_results[0].plot(labels=False, boxes=False, conf=False)

# Run YOLO pose estimation
yolo_results = model.predict(test_image)
yolo_annotated_frame = yolo_results[0].plot(labels=False, boxes=False, conf=False)

# Run face recognition
small_frame = cv2.resize(test_image, (0, 0), fx=0.5, fy=0.5)
face_locations = face_recognition.face_locations(small_frame)
face_encodings = face_recognition.face_encodings(small_frame, face_locations)

face_names = []
for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
    name = "Unknown"
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    face_names.append(name)

# Overlay face name on YOLO-annotated frame
if len(face_names) > 0:
    name_to_display = face_names[0]
    cv2.putText(yolo_annotated_frame, name_to_display, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

# Display the resulting image
cv2.imshow('Test Image', yolo_annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()