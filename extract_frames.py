import cv2
import sys
import os

def extract_frames(video_path, output_prefix, num_frames):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame indices to extract
    if total_frames <= num_frames:
        # If video has fewer frames than requested, extract all frames
        frame_indices = list(range(total_frames))
    else:
        # Otherwise, extract frames evenly spaced throughout the video
        step = total_frames / num_frames
        frame_indices = [int(i * step) for i in range(num_frames)]
    
    # Extract frames
    extracted_frames = 0
    for i, frame_idx in enumerate(frame_indices):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame at index {frame_idx}")
            continue
        
        # Save the frame
        output_path = f"{output_prefix}{i+1}.png"
        cv2.imwrite(output_path, frame)
        print(f"Extracted frame {i+1} to {output_path}")
        extracted_frames += 1
    
    # Release the video capture
    cap.release()
    
    print(f"Extracted {extracted_frames} frames from {video_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py video_file.mp4 [output_prefix] [num_frames]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "frame"
    num_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    extract_frames(video_path, output_prefix, num_frames)
