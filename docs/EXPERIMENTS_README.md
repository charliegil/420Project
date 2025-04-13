# Motion Estimation Experiments

This directory contains scripts for running experiments with different motion estimation algorithms and analyzing the results.

## Available Scripts

### 1. `run_experiments.sh`

This script runs multiple experiments with different parameters and saves the results to the `results` directory.

**Usage:**
```bash
./run_experiments.sh
```

This will run experiments with:
- Different block sizes (4, 8, 16, 32 pixels)
- Different search areas (3, 5, 7, 9, 11 pixels)
- Different algorithms (Three Step Search, Full Search, Diamond Search)
- Combined parameters for a comprehensive comparison

### 2. `analyze_results.sh`

This script analyzes the results from the experiments and generates a comprehensive report.

**Usage:**
```bash
./analyze_results.sh
```

This will generate:
- A CSV file (`results.csv`) with all experiment results
- A detailed analysis report (`analysis_report.txt`) with comparisons and recommendations

### 3. `run_experiment.sh`

This script runs a single experiment with specific parameters.

**Usage:**
```bash
./run_experiment.sh [algorithm] [block_size] [search_area] [--show-images]
```

**Parameters:**
- `algorithm`: The search algorithm to use (`three_step`, `full`, or `diamond`)
- `block_size`: The size of the blocks in pixels (e.g., 4, 8, 16, 32)
- `search_area`: The size of the search area in pixels (e.g., 3, 5, 7, 9, 11)
- `--show-images`: Optional flag to display the input, predicted, and residual images

**Examples:**
```bash
./run_experiment.sh three_step 8 7
./run_experiment.sh full 16 9 --show-images
./run_experiment.sh diamond 8 7 --show-images
```

### 4. `view_images.sh`

This script displays the images for a specific configuration.

**Usage:**
```bash
./view_images.sh [algorithm] [block_size] [search_area]
```

**Parameters:**
- `algorithm`: The search algorithm to use (`three_step`, `full`, or `diamond`)
- `block_size`: The size of the blocks in pixels
- `search_area`: The size of the search area in pixels

**Examples:**
```bash
./view_images.sh three_step 8 7
./view_images.sh full 16 9
./view_images.sh diamond 8 7
```

