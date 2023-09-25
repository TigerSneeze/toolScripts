import cv2

# Specify the path for your original video
original_video_path = "pid_vs_ideal.mp4"
savename = "pid_vs_ideal_Speedup.mp4"

# Specify the factor by which the video will be speeded up
speed_up_factor = 2.0  # 2x speed up

# Create a VideoCapture object
cap = cv2.VideoCapture(original_video_path)

# Get original video frame dimensions and fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(savename, fourcc, fps * speed_up_factor, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Write the frame to the output video
    out.write(frame)

    # Optional: display the frame
    cv2.imshow("Speeded up video", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and VideoWriter objects and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
