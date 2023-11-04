import face_recognition
import cv2
import numpy as np
import os

print("Initializing...")

video_path = 'iphone_dynamic_baseballthrow.MOV'
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('face_pose_output.mp4',
                      fourcc, 30.0, (frame_width, frame_height))

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

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

print("Starting video processing...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Reached end of video.")
        break

    if process_this_frame:
        print("Processing frame...")
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        small_frame = frame  # using original frame for now

        face_locations = face_recognition.face_locations(small_frame)
        print(f"Found {len(face_locations)} face(s) in this frame.")

        face_encodings = face_recognition.face_encodings(
            small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
        print(f"Identified faces: {face_names}")

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

    out.write(frame)

print("Releasing video objects...")
out.release()
cap.release()
