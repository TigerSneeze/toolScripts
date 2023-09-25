import cv2
import numpy as np

# Paths to the video files
video_paths = ['animation_sub_fail3.mp4', 'animation_sub_mdpso.mp4', 'animation_sub_nsga_trade.mp4', 'animation_sub_nsga_anchor.mp4']

# Read all the videos and get their dimensions
caps = [cv2.VideoCapture(path) for path in video_paths]
widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]
heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]

# Find the smallest width and height
min_width = min(widths)
min_height = min(heights)

# Create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('combined_video4.mp4', fourcc, 30.0, (2*min_width, 2*min_height))

# Initialize last frame for each video
last_frames = [None for _ in caps]

while True:
    frames = []
    ret_flags = []

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (min_width, min_height))
            last_frames[i] = resized_frame
            frames.append(last_frames[i])
        else:
            if last_frames[i] is not None:
                frames.append(last_frames[i])
            else:
                break

        ret_flags.append(ret)

    if all(flag is False for flag in ret_flags):
        break

    # Reshape the list of frames into 2x2 and then concatenate them
    grid_frames = [np.hstack(frames[i:i+2]) for i in range(0, len(frames), 2)]
    combined_frame = np.vstack(grid_frames)
    
    out.write(combined_frame)

    cv2.imshow('Combined Video', combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Add 3-second freeze at the end (90 frames)
for _ in range(90):
    out.write(combined_frame)

# Release everything
for cap in caps:
    cap.release()
out.release()
cv2.destroyAllWindows()
