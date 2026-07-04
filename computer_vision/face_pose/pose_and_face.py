import argparse
import os
import sys

import face_recognition
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_known_faces, make_writer, iter_frames

parser = argparse.ArgumentParser(description="YOLO pose estimation with face recognition.")
parser.add_argument('--video', default='iphone_dynamic_baseballthrow.MOV')
parser.add_argument('--name', default='Subject')
args = parser.parse_args()

# Initialize YOLO model
model = YOLO('yolov8n-pose.pt')

print("Initializing...")

# Initialize video capture and a writer sized to the input
cap = cv2.VideoCapture(args.video)
frame_height = int(cap.get(4))
out = make_writer(cap, 'face_pose_output.mp4')

print("Loading training images...")
known_face_encodings, known_face_names = load_known_faces('training/', args.name)

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Initialize persistent name variable and flag for face recognition
persistent_name = None
run_face_recognition = True

print("Starting video processing...")
for frame in iter_frames(cap):
    print("Processing frame...")

    # Face Recognition
    if run_face_recognition:
        face_locations = face_recognition.face_locations(frame)
        print(f"Found {len(face_locations)} face(s) in this frame.")
        
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                persistent_name = known_face_names[best_match_index]
                run_face_recognition = False  # Stop running face recognition
                break

    print(f"Identified faces: {persistent_name if persistent_name else 'None'}")

    # YOLO Pose Estimation
    yolo_results = model.predict(frame)
    frame_with_pose = yolo_results[0].plot(labels=False, boxes=False, conf=False)  # No bounding boxes

    # Write names for Face Recognition
    if persistent_name:
        cv2.putText(frame_with_pose, persistent_name, (50, frame_height - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    # Write the frame to the output video
    out.write(frame_with_pose)

print("Releasing video objects...")
out.release()
cap.release()