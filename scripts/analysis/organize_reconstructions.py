#!/usr/bin/env python3
import os
import shutil
import pandas as pd
from datetime import datetime

def create_reconstruction_directory():
    """Create a timestamped directory for reconstructed images."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "reconstructions"
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def copy_and_rename_images(df, base_dir):
    """Copy and rename reconstructed images based on algorithm configurations."""
    # Create directories for each algorithm
    for algo in df['Algorithm'].unique():
        algo_dir = os.path.join(base_dir, algo.lower().replace(' ', '_'))
        os.makedirs(algo_dir, exist_ok=True)
        
        # Get configurations for this algorithm
        algo_data = df[df['Algorithm'] == algo]
        
        for _, row in algo_data.iterrows():
            # Create configuration name
            config_name = f"block_{row['Block Size']}_search_{row['Search Area']}"
            config_dir = os.path.join(algo_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)
            
            # Define source and destination paths for each image type
            image_mappings = {
                'reconstructed_current_frame.png': 'reconstructed.png',
                'predicted_frame.png': 'predicted.png',
                'residual_frame.png': 'residual.png',
                'processed_current.png': 'current.png',
                'processed_previous.png': 'previous.png'
            }
            
            # Copy and rename images
            for src_name, dst_name in image_mappings.items():
                src_path = os.path.join('../../', src_name)
                if os.path.exists(src_path):
                    dst_path = os.path.join(config_dir, dst_name)
                    shutil.copy2(src_path, dst_path)
            
            # Create info file with metrics
            info_path = os.path.join(config_dir, 'metrics.txt')
            with open(info_path, 'w') as f:
                f.write(f"Algorithm: {algo}\n")
                f.write(f"Block Size: {row['Block Size']}\n")
                f.write(f"Search Area: {row['Search Area']}\n")
                f.write(f"Residual Metric: {row['Residual Metric']:.3f}\n")
                f.write(f"Runtime: {row['Runtime (s)']:.3f} seconds\n")

def main():
    # Read results file
    results_path = '../../results.csv'
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        exit(1)
    
    df = pd.read_csv(results_path)
    
    # Create output directory
    output_dir = create_reconstruction_directory()
    
    # Copy and organize images
    copy_and_rename_images(df, output_dir)
    
    print(f"\nReconstructions have been organized in: {output_dir}")
    print("\nDirectory structure:")
    print("reconstructions/")
    print("└── YYYYMMDD_HHMMSS/")
    for algo in df['Algorithm'].unique():
        algo_name = algo.lower().replace(' ', '_')
        print(f"    └── {algo_name}/")
        algo_data = df[df['Algorithm'] == algo]
        for _, row in algo_data.iterrows():
            config = f"block_{row['Block Size']}_search_{row['Search Area']}"
            print(f"        └── {config}/")
            print("            ├── reconstructed.png")
            print("            ├── predicted.png")
            print("            ├── residual.png")
            print("            ├── current.png")
            print("            ├── previous.png")
            print("            └── metrics.txt")

if __name__ == "__main__":
    main() 