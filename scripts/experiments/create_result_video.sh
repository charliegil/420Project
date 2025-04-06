#!/bin/bash

# Check if the required arguments are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 [output_video.mp4] [frame1.png] [frame2.png] [algorithm] [block_size] [search_area]"
    echo "  output_video.mp4: Name of the output video file"
    echo "  frame1.png, frame2.png: Input frame images"
    echo "  algorithm: three_step, full, or diamond"
    echo "  block_size: integer (e.g., 4, 8, 16, 32)"
    echo "  search_area: integer (e.g., 3, 5, 7, 9, 11)"
    exit 1
fi

# Parse arguments
output_video=$1
frame1=$2
frame2=$3
algorithm=$4
block_size=${5:-16}
search_area=${6:-7}

# Validate algorithm
if [ "$algorithm" != "three_step" ] && [ "$algorithm" != "full" ] && [ "$algorithm" != "diamond" ]; then
    echo "Error: Invalid algorithm. Must be 'three_step', 'full', or 'diamond'"
    exit 1
fi

# Set the search flag based on the algorithm
search_flag=""
if [ "$algorithm" == "three_step" ]; then
    search_flag=""
elif [ "$algorithm" == "full" ]; then
    search_flag="--full-search"
elif [ "$algorithm" == "diamond" ]; then
    search_flag="--diamond-search"
fi

# Create a temporary directory for the output frames
temp_dir=$(mktemp -d)
echo "Created temporary directory: $temp_dir"

# Run the motion estimation program with show-images flag and save the output
echo "Running motion estimation with $algorithm algorithm..."
./build/420Project --previous "$frame1" --current "$frame2" --block-size "$block_size" --search-area "$search_area" $search_flag --show-images > "$temp_dir/output.txt"

# Extract the paths of the output images from the program output
processed_previous=$(grep "Processed previous frame saved to:" "$temp_dir/output.txt" | awk '{print $6}')
processed_current=$(grep "Processed current frame saved to:" "$temp_dir/output.txt" | awk '{print $6}')
predicted_frame=$(grep "Predicted frame saved to:" "$temp_dir/output.txt" | awk '{print $5}')
residual_frame=$(grep "Residual frame saved to:" "$temp_dir/output.txt" | awk '{print $5}')
naive_residual_frame=$(grep "Naive residual frame saved to:" "$temp_dir/output.txt" | awk '{print $6}')
reconstructed_frame=$(grep "Reconstructed current frame saved to:" "$temp_dir/output.txt" | awk '{print $7}')

# Check if all images were generated
if [ -z "$processed_previous" ] || [ -z "$processed_current" ] || [ -z "$predicted_frame" ] || [ -z "$residual_frame" ] || [ -z "$naive_residual_frame" ] || [ -z "$reconstructed_frame" ]; then
    echo "Error: Some output images were not generated"
    exit 1
fi

# Create a Python script to generate the video
cat > "$temp_dir/create_video.py" << 'EOF'
import cv2
import sys
import os

def create_video(output_path, frame_paths, fps=30, duration_per_frame=2):
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each frame to the video, repeating each frame for the specified duration
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Could not read frame {frame_path}")
            continue
        
        # Add text to the frame
        frame_name = os.path.basename(frame_path)
        cv2.putText(frame, frame_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write the frame multiple times to make it visible longer
        for _ in range(fps * duration_per_frame):
            out.write(frame)
    
    # Release the video writer
    out.release()
    print(f"Video created successfully: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_video.py output_video.mp4 frame1.png frame2.png [frame3.png] [frame4.png]")
        sys.exit(1)
    
    output_path = sys.argv[1]
    frame_paths = sys.argv[2:]
    create_video(output_path, frame_paths)
EOF

# Make the Python script executable
chmod +x "$temp_dir/create_video.py"

# Run the Python script to create the video
python3 "$temp_dir/create_video.py" "$output_video" "$frame1" "$frame2" "$processed_previous" "$processed_current" "$predicted_frame" "$residual_frame" "$naive_residual_frame" "$reconstructed_frame"

# Check if the video was created successfully
if [ $? -eq 0 ]; then
    echo "Video creation completed successfully"
    echo "Video saved to: $output_video"
else
    echo "Error: Failed to create video"
    exit 1
fi

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf "$temp_dir" 