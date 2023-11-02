import face_recognition  # Library for face recognition
import cv2  # OpenCV library
import numpy as np  # Library for numerical operations
import os  # Library for operating system dependent functionality

print("Initializing...")

# Path to the video file to be processed
video_path = 'calibration1.mp4'
# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)
# Get the width and height of frames in the video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('face_identification_output.mp4',
                      fourcc, 30.0, (frame_width, frame_height))

print("Loading training images...")
# Initialize lists to store face encodings and names
known_face_encodings = []
known_face_names = []

# Load training images from the 'training' directory
if os.path.exists('training/'):
    for filename in os.listdir('training/'):
        if filename.endswith('.jpg'):
            print(f"Processing {filename}...")
            image_path = os.path.join('training/', filename)
            # Load an image file to find face locations
            image = face_recognition.load_image_file(image_path)
            # Get face encodings for any faces in the uploaded image
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                face_encoding = encodings[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append("Clayton Thompson")
            else:
                print(f"No faces found in {filename}")

print(f"Loaded {len(known_face_encodings)} face encodings.")

# Initialize lists to store face locations, encodings, and names
face_locations = []
face_encodings = []
face_names = []
# Initialize variable to control frame processing
process_this_frame = True

print("Starting video processing...")

# Loop over frames from the video file stream
while cap.isOpened():
    # Read the next frame from the file
    ret, frame = cap.read()
    # If the frame was not grabbed, then we have reached the end of the stream
    if not ret:
        print("Reached end of video.")
        break

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
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    out.write(frame)

print("Releasing video objects...")
# Release the file pointers
out.release()
cap.release()
