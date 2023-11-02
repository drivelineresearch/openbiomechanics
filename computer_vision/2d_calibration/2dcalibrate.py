import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def count_images_in_directory(directory_path, camera_ids):
    min_success_frames = float('inf')

    for camera_id in camera_ids:
        success_images = glob.glob(f"{directory_path}/camera_{camera_id}_frame_*_success.png")
        num_success = len(success_images)

        if num_success < min_success_frames:
            min_success_frames = num_success

    return min_success_frames

def calibrate_camera(frames_dict, checkerboard_size, camera_id):
    
    print(f"Starting calibration for camera {camera_id}...")

    obj_points = []
    img_points = []

    frames = frames_dict[camera_id]

    objp = np.zeros((np.prod(checkerboard_size), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * block_size

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if ret:
            img_points.append(corners)
            obj_points.append(objp)

            # Save the corners of the first good frame only
            cv2.drawChessboardCorners(frame, checkerboard_size, corners, ret)
            cv2.imwrite(f'debug_images/found_corners_camera_{camera_id}_frame_{i}.png', frame)
            print(f"Found corners for frame {i} of camera {camera_id}.")
            break  # Stop after finding the first good frame

    print(f"Performing camera calibration for camera {camera_id}...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    print(f"Calibration for camera {camera_id} completed.")
    return mtx, dist, obj_points, img_points

def load_and_process_frames(folder, camera_ids, max_frames=75):
    frames_dict = {}
    for camera_id in camera_ids:
        frames = []
        image_files = sorted(glob.glob(f"{folder}/camera_{camera_id}_frame_*_success.png"))[:max_frames]
        print(f"Found {len(image_files)} probable successful frames for camera {camera_id}. Using up to {max_frames}.")
        for image_file in image_files:
            frame = cv2.imread(image_file)
            frames.append(frame)
        frames_dict[camera_id] = frames
    return frames_dict

def plot_single_camera(camera_id):
    frames = frames_dict[camera_id]
       
    # Initialize 2D points container
    all_points2D = []

    xlims = [450, 900]
    ylims = [200, 600]

    markers = {'1': 'o', '2': 'x', '3': 's', '4': '^', '5': 'p', '6': 'h', '7': 'D', '8': '*'}
    colors = {'1': 'r', '2': 'g', '3': 'b', '4': 'c', '5': 'm', '6': 'y', '7': 'k', '8': 'orange'}

    for i, img in enumerate(frames):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            all_points2D.append((corners, camera_id))

    # Plot and save each camera separately
    for corners, cam in all_points2D:
        plt.figure()
        plt.scatter(corners[:, 0, 0], corners[:, 0, 1], marker=markers[cam], c=colors[cam], label=f'Camera {cam}')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title(f'2D Chessboard Corners (Camera {cam})')
        plt.legend()
        plt.savefig(f'figures/2D_chessboard_corners_camera_{cam}.png')

if __name__ == "__main__":
    # Initialize parameters
    checkerboard_size = (7, 4)
    block_size = 100  # mm

    # Load frames for all cameras
    camera_ids_to_use = ['1', '2', '3', '4', '5', '6', '7', '8']  # Add all camera IDs you want to use
    
    directory_path = "multiprocessed_frames"
    max_frames = count_images_in_directory(directory_path, camera_ids_to_use)
    
    if max_frames > 50:
        max_frames = 50
    
    print(f"Maximum number of frames to be used for calibration: {max_frames}")

    frames_dict = load_and_process_frames(directory_path, camera_ids_to_use, max_frames=max_frames)

    # 2D plots
    print("running plot_single_camera and calibrate_camera from main")
    for camera_id in camera_ids_to_use:
        print(f"Processing camera id: {camera_id}")
        plot_single_camera(camera_id)
        calibrate_camera(frames_dict, checkerboard_size, camera_id)

    
