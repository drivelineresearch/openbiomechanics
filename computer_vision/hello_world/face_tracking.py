import os
import sys

import face_recognition  # Library for face recognition
import cv2  # OpenCV library
import numpy as np  # Library for numerical operations

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_known_faces, make_writer, iter_frames, draw_face_box

print("Initializing...")

# Path to the video file to be processed
video_path = 'calibration1.mp4'
# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)
# Create a VideoWriter sized to the input video
out = make_writer(cap, 'face_identification_output.mp4')

print("Loading training images...")
# Load face encodings from the 'training' directory
known_face_encodings, known_face_names = load_known_faces('training/', "Subject")

# Initialize lists to store face locations, encodings, and names
face_locations = []
face_encodings = []
face_names = []
# Initialize variable to control frame processing
process_this_frame = True

print("Starting video processing...")

# Loop over frames from the video file stream
for frame in iter_frames(cap):
    # Only process every other frame of video to save time
    if process_this_frame:
        print("Processing frame...")
        # Resize frame of video for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        small_frame = frame  # using original frame for now

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        print(f"Found {len(face_locations)} face(s) in this frame.")

        face_encodings = face_recognition.face_encodings(
            small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
        print(f"Identified faces: {face_names}")

    # Switch to not process the next frame
    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        draw_face_box(frame, top, right, bottom, left, name)

    # Write the resulting image to the output video file
    out.write(frame)

print("Releasing video objects...")
# Release the file pointers
out.release()
cap.release()
