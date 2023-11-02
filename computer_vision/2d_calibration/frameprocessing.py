import os
import cv2
import glob
import threading

def extract_and_process_frames(video_file, checkerboard_size, output_folder):
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print(f"Failed to open video file: {video_file}")
        return
    
    camera_id = os.path.basename(video_file).split('_')[-1].split('.')[0]  # Extracting 'x' from 'checkerboard_static_x.mp4'
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        frame_filename = f"{output_folder}/camera_{camera_id}_frame_{frame_count}.png"
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        # Rename file based on success or failure
        if ret:
            cv2.imwrite(frame_filename.replace(".png", "_success.png"), frame)
        else:
            cv2.imwrite(frame_filename.replace(".png", "_fail.png"), frame)
        
        frame_count += 1
    
    cap.release()

# Parameters
checkerboard_size = (7, 4)  # (rows, cols) - Number of inner corners per a chessboard row and column.
output_folder = "multiprocessed_frames"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each video file
video_files = glob.glob("videos/checkerboard_static_*.mp4")

threads = []
for video_file in video_files:
    thread = threading.Thread(target=extract_and_process_frames, args=(video_file, checkerboard_size, output_folder))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("Frame extraction and processing complete.")
