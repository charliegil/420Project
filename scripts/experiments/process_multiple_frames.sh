#!/bin/bash

# Check if the required arguments are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 [output_video.mp4] [frame1.png] [frame2.png] [frame3.png] ... [algorithm] [block_size] [search_area]"
    echo "  output_video.mp4: Name of the output video file"
    echo "  frame1.png, frame2.png, etc.: Input frame images"
    echo "  algorithm: three_step, full, or diamond"
    echo "  block_size: integer (e.g., 4, 8, 16, 32)"
    echo "  search_area: integer (e.g., 3, 5, 7, 9, 11)"
    exit 1
fi

# Parse arguments
output_video=$1
shift

# Find the algorithm parameter (it should be one of: three_step, full, diamond)
algorithm=""
block_size=16
search_area=7

# Extract algorithm, block_size, and search_area from the arguments
frames=()
for arg in "$@"; do
    if [ "$arg" == "three_step" ] || [ "$arg" == "full" ] || [ "$arg" == "diamond" ]; then
        algorithm=$arg
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        if [ -z "$block_size" ] || [ "$block_size" == "16" ]; then
            block_size=$arg
        else
            search_area=$arg
        fi
    else
        frames+=("$arg")
    fi
done

# Validate algorithm
if [ -z "$algorithm" ]; then
    echo "Error: Algorithm not specified. Must be 'three_step', 'full', or 'diamond'"
    exit 1
fi

# Validate that we have at least 2 frames
if [ ${#frames[@]} -lt 2 ]; then
    echo "Error: At least 2 frames are required"
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

# Process each pair of consecutive frames
all_output_frames=()
for ((i=0; i<${#frames[@]}-1; i++)); do
    frame1=${frames[$i]}
    frame2=${frames[$i+1]}
    
    echo "Processing frames $((i+1)) and $((i+2)): $frame1 and $frame2"
    
    # Run the motion estimation program with show-images flag and save the output
    ./build/420Project --previous "$frame1" --current "$frame2" --block-size "$block_size" --search-area "$search_area" $search_flag --show-images > "$temp_dir/output_$i.txt"
    
    # Extract the paths of the output images from the program output
    processed_previous=$(grep "Processed previous frame saved to:" "$temp_dir/output_$i.txt" | awk '{print $6}')
    processed_current=$(grep "Processed current frame saved to:" "$temp_dir/output_$i.txt" | awk '{print $6}')
    predicted_frame=$(grep "Predicted frame saved to:" "$temp_dir/output_$i.txt" | awk '{print $5}')
    residual_frame=$(grep "Residual frame saved to:" "$temp_dir/output_$i.txt" | awk '{print $5}')
    naive_residual_frame=$(grep "Naive residual frame saved to:" "$temp_dir/output_$i.txt" | awk '{print $6}')
    reconstructed_frame=$(grep "Reconstructed current frame saved to:" "$temp_dir/output_$i.txt" | awk '{print $7}')
    
    # Check if all images were generated
    if [ -z "$processed_previous" ] || [ -z "$processed_current" ] || [ -z "$predicted_frame" ] || [ -z "$residual_frame" ] || [ -z "$naive_residual_frame" ] || [ -z "$reconstructed_frame" ]; then
        echo "Error: Some output images were not generated for frames $((i+1)) and $((i+2))"
        continue
    fi
    
    # Add the output frames to the list
    all_output_frames+=("$frame1" "$frame2" "$processed_previous" "$processed_current" "$predicted_frame" "$residual_frame" "$naive_residual_frame" "$reconstructed_frame")
done

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
python3 "$temp_dir/create_video.py" "$output_video" "${all_output_frames[@]}"

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