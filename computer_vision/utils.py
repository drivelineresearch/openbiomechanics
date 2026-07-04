"""Shared OpenCV / face_recognition helpers for the computer_vision examples.

The example scripts live in subdirectories (hello_world/, face_pose/,
face_tracking/), so they add this directory to sys.path before importing:

    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import load_known_faces, make_writer, iter_frames, draw_face_box
"""
import os

import cv2
import face_recognition


def load_known_faces(training_dir, name):
    """Encode every .jpg in training_dir, labeling each face with name.

    Returns parallel (encodings, names) lists.
    """
    encodings = []
    names = []
    if os.path.exists(training_dir):
        for filename in os.listdir(training_dir):
            if filename.endswith('.jpg'):
                print(f"Processing {filename}...")
                image = face_recognition.load_image_file(os.path.join(training_dir, filename))
                found = face_recognition.face_encodings(image)
                if len(found) > 0:
                    encodings.append(found[0])
                    names.append(name)
                else:
                    print(f"No faces found in {filename}")
    print(f"Loaded {len(encodings)} face encodings.")
    return encodings, names


def make_writer(cap, path, fps=30.0):
    """Create an mp4v VideoWriter at path sized to cap's frame dimensions."""
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def iter_frames(cap):
    """Yield frames from an open VideoCapture until the stream ends."""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video.")
            break
        yield frame


def draw_face_box(frame, top, right, bottom, left, name):
    """Draw a red box around a face with a filled name label beneath it."""
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
