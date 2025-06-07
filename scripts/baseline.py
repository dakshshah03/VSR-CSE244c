import cv2
import os
import argparse
import glob
from tqdm import tqdm

def upscale_frames(input_dir, output_dir, scale_factor):
    """
    Upscale all frames from input directory and save to output directory
    
    Args:
        input_dir (str): Directory containing input frames
        output_dir (str): Directory where upscaled frames will be saved
        scale_factor (float): Scale factor for upscaling
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all image files in input directory
    input_frames = sorted(glob.glob(os.path.join(input_dir, '*.png')) + 
                         glob.glob(os.path.join(input_dir, '*.jpg')))
    
    if not input_frames:
        print(f"No image frames found in {input_dir}")
        return
    
    # Process each frame
    for i, frame_path in enumerate(tqdm(input_frames, desc="Upscaling frames")):
        # Read the frame
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
            
        # Get original dimensions
        height, width = frame.shape[:2]
        
        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize frame using bicubic interpolation
        upscaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Generate output filename
        frame_name = os.path.basename(frame_path)
        base_name, ext = os.path.splitext(frame_name)
        output_path = os.path.join(output_dir, frame_name)
        
        # Write the upscaled frame
        cv2.imwrite(output_path, upscaled_frame)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Upscale video frames using bicubic interpolation')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing frames')
    parser.add_argument('--output', '-o', required=True, help='Output directory for upscaled frames')
    parser.add_argument('--scale', '-s', type=float, default=4.0, 
                        help='Scale factor for upscaling (default: 4.0)')
    
    args = parser.parse_args()
    
    # Call the upscale function
    upscale_frames(args.input, args.output, args.scale)
    
    print(f"Upscaling complete. Upscaled frames saved to {args.output}")

if __name__ == "__main__":
    main()