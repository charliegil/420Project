# Motion Estimation Project

This project implements and compares different motion estimation algorithms for video compression. It includes implementations of Full Search, Three Step Search, and Diamond Search algorithms, along with tools for analysis and visualization.

## Project Structure

```
420Project/
├── src/                    # Source code
│   ├── main.cpp           # Main program implementation
│   └── CMakeLists.txt     # CMake configuration for source files
│
├── scripts/               # Scripts for various tasks
│   ├── analysis/         # Analysis and visualization scripts
│   │   ├── run_experiments.py    # Run experiments with different parameters
│   │   └── visualize_results.py  # Generate plots from experiment results
│   │
│   ├── experiments/      # Experiment management scripts
│   │   ├── run_experiment.sh     # Run single experiment
│   │   └── analyze_results.sh    # Analyze experiment results
│   │
│   └── comparison_utils/ # Utilities for comparing algorithms
│       └── compare_algorithms.sh # Compare different algorithms
│
├── data/                 # Data directory
│   └── frames/          # Input frames for testing
│
├── docs/                # Documentation
│   └── EXPERIMENTS_README.md  # Detailed experiment documentation
│
├── build/               # Build directory (created during build)
├── results/             # Results directory (created during experiments)
└── CMakeLists.txt       # Main CMake configuration
```

## Prerequisites

- CMake (version 3.10 or higher)
- OpenCV (version 4.x)
- Python 3.x with required packages:
  - pandas
  - matplotlib
  - numpy

## Building the Project

1. Create a build directory:
```bash
mkdir build && cd build
```

2. Configure and build:
```bash
cmake ..
make
```

## Running the Program


### Running Experiments

#### 1. Run Single Experiment
```bash
./scripts/experiments/run_experiment.sh [algorithm] [block_size] [search_area] [--show-images]
```
Parameters:
- `algorithm`: three_step, full, or diamond
- `block_size`: integer (e.g., 4, 8, 16, 32, 64)
- `search_area`: integer (e.g., 7, 11, 14, 21, 28, 35)
- `--show-images`: optional flag to display images

#### 2. Run Comprehensive Experiments
```bash
python3 scripts/analysis/run_experiments.py
```
This will test all combinations of:
- Algorithms: Full Search, Three Step Search, Diamond Search
- Block sizes: 4, 8, 16, 32, 64
- Search areas: 7, 11, 14, 21, 28, 35

Results will be saved in `experiment_results/[timestamp]/` with:
- Reconstructed frames
- Execution logs
- Summary CSV file

#### 3. Visualize Results
```bash
python3 scripts/analysis/visualize_results.py
```
This will generate plots in the `plots/` directory showing:
- Performance vs quality trade-offs
- Block size comparisons
- Search area comparisons
- Algorithm comparisons

### Direct Program Usage

While you can run the program directly:
```bash
./build/motion_estimation --previous frame1.png --current frame2.png --block-size 16 --search-area 7 [--full-search|--diamond-search]
```

Using the scripts is recommended because they:
1. Handle all parameter combinations
2. Organize results automatically
3. Generate analysis and visualizations
4. Ensure consistent experiment setup

## Output Files

The program generates several output files for each experiment:
- `processed_previous.png`: Processed previous frame
- `processed_current.png`: Processed current frame
- `predicted_frame.png`: Predicted frame using motion vectors
- `residual_frame.png`: Difference between predicted and current frame
- `naive_residual_frame.png`: Difference between previous and current frame
- `reconstructed_current_frame.png`: Reconstructed current frame

## Analysis

After running experiments, you can find:
- `results.csv`: Summary of all experiments
- `analysis_report.txt`: Detailed analysis of results
- Various PNG files showing comparisons and visualizations

## Documentation

For more detailed information about experiments and analysis, see:
- `docs/EXPERIMENTS_README.md`: Detailed experiment documentation
- `docs/ALGORITHMS.md`: Algorithm descriptions and comparisons 