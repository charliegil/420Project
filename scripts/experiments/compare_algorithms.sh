#!/bin/bash

# Check if the required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 [output_video.mp4] [frame1.png] [frame2.png] [frame3.png] ... [block_size] [search_area]"
    echo "  output_video.mp4: Name of the output video file"
    echo "  frame1.png, frame2.png, etc.: Input frame images"
    echo "  block_size: integer (e.g., 4, 8, 16, 32)"
    echo "  search_area: integer (e.g., 3, 5, 7, 9, 11)"
    exit 1
fi

# Parse arguments
output_video=$1
shift

# Find the block_size and search_area parameters
block_size=16
search_area=7

# Extract block_size and search_area from the arguments
frames=()
for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        if [ -z "$block_size" ] || [ "$block_size" == "16" ]; then
            block_size=$arg
        else
            search_area=$arg
        fi
    else
        frames+=("$arg")
    fi
done

# Validate that we have at least 2 frames
if [ ${#frames[@]} -lt 2 ]; then
    echo "Error: At least 2 frames are required"
    exit 1
fi

# Create a temporary directory for the output frames
temp_dir=$(mktemp -d)
echo "Created temporary directory: $temp_dir"

# Function to rename output files with algorithm prefix
rename_output_files() {
    local algorithm=$1
    local prefix="${algorithm}_"
    mv processed_previous.png "${prefix}processed_previous.png"
    mv processed_current.png "${prefix}processed_current.png"
    mv predicted_frame.png "${prefix}predicted_frame.png"
    mv residual_frame.png "${prefix}residual_frame.png"
    mv naive_residual_frame.png "${prefix}naive_residual_frame.png"
    mv reconstructed_current_frame.png "${prefix}reconstructed_current_frame.png"
}

# Process frames with Three Step Search
echo "Processing frames with Three Step Search..."
./build/420Project --previous "${frames[0]}" --current "${frames[1]}" --block-size "$block_size" --search-area "$search_area" --show-images > "$temp_dir/three_step_output.txt"
rename_output_files "three_step"

# Process frames with Full Search
echo "Processing frames with Full Search..."
./build/420Project --previous "${frames[0]}" --current "${frames[1]}" --block-size "$block_size" --search-area "$search_area" --full-search --show-images > "$temp_dir/full_output.txt"
rename_output_files "full"

# Process frames with Diamond Search
echo "Processing frames with Diamond Search..."
./build/420Project --previous "${frames[0]}" --current "${frames[1]}" --block-size "$block_size" --search-area "$search_area" --diamond-search --show-images > "$temp_dir/diamond_output.txt"
rename_output_files "diamond"

# Create a list of frames to include in the video
all_output_frames=(
    "${frames[0]}"
    "${frames[1]}"
    "three_step_processed_previous.png"
    "three_step_processed_current.png"
    "three_step_predicted_frame.png"
    "three_step_residual_frame.png"
    "three_step_naive_residual_frame.png"
    "three_step_reconstructed_current_frame.png"
    "full_predicted_frame.png"
    "full_residual_frame.png"
    "full_reconstructed_current_frame.png"
    "diamond_predicted_frame.png"
    "diamond_residual_frame.png"
    "diamond_reconstructed_current_frame.png"
)

# Debug: Print the list of frames to be included in the video
echo "Frames to be included in the video:"
for frame in "${all_output_frames[@]}"; do
    echo "  $frame"
    if [ ! -f "$frame" ]; then
        echo "    Warning: File not found"
    fi
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
        if not frame_path or frame_path.strip() == "":
            print(f"Warning: Empty frame path, skipping")
            continue
            
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