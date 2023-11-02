import face_recognition
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Initialize YOLO model
model = YOLO('yolov8n-pose.pt')

print("Initializing...")

# Initialize video capture
video_path = 'iphone_dynamic_baseballthrow.MOV'
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('face_pose_output.mp4', fourcc, 30.0, (frame_width, frame_height))

print("Loading training images...")
known_face_encodings = []
known_face_names = []

# Load training images
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

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Initialize persistent name variable and flag for face recognition
persistent_name = None
run_face_recognition = True

print("Starting video processing...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Reached end of video.")
        break

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