import os
import cv2
from glob import glob
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Merge frames into a video file.')
    parser.add_argument('--input_dir', '-i', type=str, required=True, 
                        help='Directory containing input frames (PNG files)')
    parser.add_argument('--output_file', '-o', type=str, required=True,
                        help='Output video file path')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the output video (default: 30)')
    args = parser.parse_args()
    
    image_folder = args.input_dir

    # Get sorted list of image files
    images = sorted(glob(os.path.join(image_folder, '*.png')))
    
    if not images:
        print(f"Error: No PNG files found in {image_folder}")
        return
    
    # Read the first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define output video settings
    output_path = args.output_file
    fps = args.fps  # frames per second

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' or 'avc1'
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame to the video
    print(f"Processing {len(images)} frames...")
    for i, image in enumerate(images):
        frame = cv2.imread(image)
        video.write(frame)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(images)} frames")

    # Release the video writer
    video.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()