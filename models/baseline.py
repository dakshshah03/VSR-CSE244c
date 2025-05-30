import cv2

# Load input video
input_path = 'input_video.mp4'
cap = cv2.VideoCapture(input_path)

# Get original video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define scale factor
scale_factor = 2  # upscale by 2x
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter('output_upscaled.mp4', fourcc, fps, (new_width, new_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame using bicubic interpolation
    upscaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Write to output video
    out.write(upscaled_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
