# Results Directory

This directory contains the results of motion estimation experiments and analyses.

## Directory Structure

```
results/
├── analysis/              # Analysis results
│   ├── *.png             # Generated plots and visualizations
│   ├── results.csv       # Summary of experiment results
│   └── analysis_report.txt # Detailed analysis report
│
└── experiment_results/    # Experiment outputs
    └── YYYYMMDD_HHMMSS/  # Timestamped experiment runs
        ├── algorithm_block_size_search_area/
        │   ├── frames/   # Generated frames
        │   └── execution_log.txt
        └── experiment_summary.csv
```

## Contents

### Analysis Directory
- `*.png`: Various plots showing algorithm comparisons, performance metrics, etc.
- `results.csv`: Summary of all experiment results in CSV format
- `analysis_report.txt`: Detailed analysis of experiment results

### Experiment Results
Each experiment run creates a timestamped directory containing:
- Subdirectories for each algorithm/parameter combination
- Generated frames for each configuration
- Execution logs
- Summary CSV file

## Notes
- This directory is excluded from version control (see .gitignore)
- Results are generated automatically by the experiment scripts
- Old results may be cleaned up periodically to save space 