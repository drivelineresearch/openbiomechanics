import os
import sys

import face_recognition
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_known_faces, make_writer, iter_frames, draw_face_box

print("Initializing...")

cap = cv2.VideoCapture('calibration1.mp4')
out = make_writer(cap, 'face_identification_output.mp4')

print("Loading training images...")
known_face_encodings, known_face_names = load_known_faces('training/', "Subject")

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

print("Starting video processing...")

for frame in iter_frames(cap):
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
        draw_face_box(frame, top, right, bottom, left, name)

    out.write(frame)

print("Releasing video objects...")
out.release()
cap.release()
