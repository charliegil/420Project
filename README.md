# Motion Estimation Project

This project implements and compares different motion estimation algorithms for video processing.

## Project Structure

```
.
├── src/                    # Source code
│   ├── main.cpp           # Main program
│   └── motion_estimation/ # Motion estimation algorithms
├── scripts/               # Scripts for running experiments
│   └── experiments/
│       ├── run_experiment.sh    # Run single experiment
│       └── run_experiments.sh   # Run comprehensive experiments
├── data/                  # Input data
│   └── frames/           # Video frames
├── docs/                 # Documentation
├── results/              # Experiment results
└── build/               # Build directory
```

## Prerequisites

- CMake 3.10 or higher
- OpenCV 4.x
- Python 3.x with required packages:
  ```bash
  pip install pandas matplotlib numpy
  ```

## Building the Project

1. Install OpenCV if not already installed:
   - macOS:
     ```bash
     brew install opencv
     ```
   - Ubuntu/Debian:
     ```bash
     sudo apt-get install libopencv-dev
     ```

2. Install required Python packages:
   ```bash
   pip install pandas matplotlib numpy
   ```

3. Build the project:
   ```bash
   # Remove any existing build directory
   rm -rf build
   
   # Create and enter build directory
   mkdir build
   cd build
   
   # Configure and build
   cmake ..
   make
   ```

## Running Experiments

### Single Experiment
```bash
./scripts/experiments/run_experiment.sh <algorithm> <block_size> <search_area>
```
Example:
```bash
./scripts/experiments/run_experiment.sh three_step 16 8
```

### Comprehensive Experiments
```bash
./scripts/experiments/run_experiments.sh
```
This will run experiments with various combinations of:
- Algorithms: Three Step Search, Diamond Search, Full Search
- Block sizes: 4, 8, 16, 32
- Search areas: 4, 8, 16, 32

### Direct Program Usage
```bash
./build/motion_estimation <algorithm> <block_size> <search_area> <frame1> <frame2>
```
Example:
```bash
./build/motion_estimation three_step 16 8 data/frames/frame1.png data/frames/frame2.png
```

## Troubleshooting

If you encounter build issues:
1. Ensure OpenCV is properly installed
2. Set OpenCV_DIR if needed:
   ```bash
   export OpenCV_DIR=/path/to/opencv/build
   ```
3. Check CMake output for any missing dependencies
