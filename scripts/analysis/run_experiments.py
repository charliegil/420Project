#!/Users/karim/miniforge3/bin/python
import os
import subprocess
import pandas as pd
from datetime import datetime
import shutil

def create_output_directory():
    """Create timestamped directories for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "experiment_results"
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def run_experiment(executable_path, algorithm, block_size, search_area, output_dir):
    """Run a single experiment with given parameters."""
    # Create directory for this configuration
    config_name = f"{algorithm.lower().replace(' ', '_')}_block_{block_size}_search_{search_area}"
    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    # Create frames directory for this configuration
    frames_dir = os.path.join(config_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Set the search flag based on the algorithm
    search_flag = ""
    if algorithm == "Full Search":
        search_flag = "--full-search"
    elif algorithm == "Diamond Search":
        search_flag = "--diamond-search"
    
    # Run the executable with parameters
    cmd = [
        executable_path,
        "--previous", "../../data/frames/frame1.png",
        "--current", "../../data/frames/frame2.png",
        "--block-size", str(block_size),
        "--search-area", str(search_area),
        "--show-images"
    ]
    
    if search_flag:
        cmd.append(search_flag)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Save command output
        with open(os.path.join(config_dir, "execution_log.txt"), "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
        
        # Copy the generated frames to the output directory
        for frame in ["processed_previous.png", "processed_current.png", 
                     "predicted_frame.png", "residual_frame.png", 
                     "naive_residual_frame.png", "reconstructed_current_frame.png"]:
            if os.path.exists(frame):
                shutil.copy2(frame, os.path.join(frames_dir, frame))
        
        return {
            'Algorithm': algorithm,
            'Block Size': block_size,
            'Search Area': search_area,
            'Exit Code': result.returncode,
            'Output Directory': config_dir,
            'Frames Directory': frames_dir
        }
    except Exception as e:
        print(f"Error running experiment: {e}")
        return None

def main():
    # Define parameter combinations to test
    algorithms = ['Full Search', 'Three Step Search', 'Diamond Search']
    block_sizes = [4, 8, 16, 32, 64]  # Added larger block size
    search_areas = [7, 11, 14, 21, 28, 35]  # Added larger search areas
    
    # Find the executable
    executable = "../../build/motion_estimation"
    if not os.path.exists(executable):
        print("Error: motion_estimation executable not found in build directory")
        print("Please build the project first")
        exit(1)
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"\nRunning experiments, results will be saved in: {output_dir}")
    print(f"Testing {len(algorithms)} algorithms with {len(block_sizes)} block sizes and {len(search_areas)} search areas")
    print(f"Total combinations: {len(algorithms) * len(block_sizes) * len(search_areas)}")
    
    # Run experiments
    results = []
    total_experiments = len(algorithms) * len(block_sizes) * len(search_areas)
    current = 0
    
    for algo in algorithms:
        for block in block_sizes:
            for search in search_areas:
                current += 1
                print(f"\nRunning experiment {current}/{total_experiments}")
                print(f"Algorithm: {algo}")
                print(f"Block Size: {block}")
                print(f"Search Area: {search}")
                
                result = run_experiment(executable, algo, block, search, output_dir)
                if result:
                    results.append(result)
    
    # Save experiment summary
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(output_dir, "experiment_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print("\nExperiments completed!")
    print(f"Results saved in: {output_dir}")
    print("\nSummary of experiments:")
    print(f"Total configurations tested: {len(results)}")
    print(f"Summary saved to: {summary_path}")
    print("\nFor each configuration, you can find:")
    print("- Reconstructed frames in the 'frames' subdirectory")
    print("- Execution logs in 'execution_log.txt'")
    print("- All results are organized in timestamped directories under 'experiment_results'")

if __name__ == "__main__":
    main() 