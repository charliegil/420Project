#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
import glob

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Read all result files
def read_result_files():
    results = []
    project_root = get_project_root()
    result_files = glob.glob(os.path.join(project_root, 'results', '*.txt'))
    
    if not result_files:
        print("Error: No result files found in the results directory.")
        return None
    
    for file in result_files:
        try:
            with open(file, 'r') as f:
                content = f.readlines()
                
            # Extract information from filename
            filename = os.path.basename(file)
            parts = filename.replace('.txt', '').split('_')
            
            # Handle "three_step" algorithm name
            if parts[0] == 'three':
                algorithm = 'three_step'
                block_size = int(parts[2].replace('block', ''))
                search_area = int(parts[3].replace('search', ''))
            else:
                algorithm = parts[0]
                block_size = int(parts[1].replace('block', ''))
                search_area = int(parts[2].replace('search', ''))
            
            # Extract metrics
            residual_metric = float([line for line in content if 'Residual Metric:' in line][0].split(':')[1].strip())
            naive_residual = float([line for line in content if 'Naive Residual Metric:' in line][0].split(':')[1].strip())
            runtime = float([line for line in content if 'Runtime:' in line][0].split(':')[1].strip().replace(' seconds', ''))
            
            results.append({
                'Algorithm': algorithm,
                'Block Size': block_size,
                'Search Area': search_area,
                'Residual Metric': residual_metric,
                'Naive Residual Metric': naive_residual,
                'Runtime (s)': runtime
            })
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def generate_algorithm_specific_plots(df, results_dir):
    # Create directory for algorithm-specific plots
    algo_dir = os.path.join(results_dir, 'algorithm_analysis')
    os.makedirs(algo_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # For each algorithm
    for algo in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algo]
        
        # 1. Search Area vs Residual Metric for different block sizes
        plt.figure(figsize=(12, 6))
        for block_size in sorted(algo_data['Block Size'].unique()):
            block_data = algo_data[algo_data['Block Size'] == block_size]
            plt.plot(block_data['Search Area'], block_data['Residual Metric'], 
                    marker='o', label=f'Block Size {block_size}')
        
        plt.title(f'Search Area vs Residual Metric - {algo.capitalize()} Search')
        plt.xlabel('Search Area')
        plt.ylabel('Residual Metric')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(algo_dir, f'{algo}_search_area_effect.png'))
        plt.close()
        
        # 2. Block Size vs Residual Metric for different search areas
        plt.figure(figsize=(12, 6))
        for search_area in sorted(algo_data['Search Area'].unique()):
            search_data = algo_data[algo_data['Search Area'] == search_area]
            plt.plot(search_data['Block Size'], search_data['Residual Metric'], 
                    marker='o', label=f'Search Area {search_area}')
        
        plt.title(f'Block Size vs Residual Metric - {algo.capitalize()} Search')
        plt.xlabel('Block Size')
        plt.ylabel('Residual Metric')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(algo_dir, f'{algo}_block_size_effect.png'))
        plt.close()
        
        # 3. Runtime vs Quality scatter plot
        plt.figure(figsize=(12, 6))
        scatter = plt.scatter(algo_data['Runtime (s)'], algo_data['Residual Metric'],
                            c=algo_data['Block Size'], cmap='viridis',
                            s=algo_data['Search Area']*20)
        plt.colorbar(scatter, label='Block Size')
        plt.title(f'Runtime vs Quality - {algo.capitalize()} Search\n(Point size = Search Area)')
        plt.xlabel('Runtime (s)')
        plt.ylabel('Residual Metric')
        plt.xscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(algo_dir, f'{algo}_runtime_quality.png'))
        plt.close()
        
        # 4. Heatmap of Residual Metric
        pivot_table = algo_data.pivot_table(
            values='Residual Metric',
            index='Block Size',
            columns='Search Area',
            aggfunc='mean'
        )
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title(f'Residual Metric Heatmap - {algo.capitalize()} Search')
        plt.savefig(os.path.join(algo_dir, f'{algo}_heatmap.png'))
        plt.close()

def generate_comparison_plots(df, results_dir):
    # Create directory for comparison plots
    comp_dir = os.path.join(results_dir, 'comparison_analysis')
    os.makedirs(comp_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # 1. Search Area effect comparison
    plt.figure(figsize=(15, 8))
    for algo in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algo]
        mean_residual = algo_data.groupby('Search Area')['Residual Metric'].mean()
        plt.plot(mean_residual.index, mean_residual.values, marker='o', label=algo)
    
    plt.title('Average Residual Metric vs Search Area by Algorithm')
    plt.xlabel('Search Area')
    plt.ylabel('Average Residual Metric')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(comp_dir, 'search_area_comparison.png'))
    plt.close()
    
    # 2. Block Size effect comparison
    plt.figure(figsize=(15, 8))
    for algo in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algo]
        mean_residual = algo_data.groupby('Block Size')['Residual Metric'].mean()
        plt.plot(mean_residual.index, mean_residual.values, marker='o', label=algo)
    
    plt.title('Average Residual Metric vs Block Size by Algorithm')
    plt.xlabel('Block Size')
    plt.ylabel('Average Residual Metric')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(comp_dir, 'block_size_comparison.png'))
    plt.close()
    
    # 3. Runtime-Quality trade-off comparison
    plt.figure(figsize=(15, 8))
    for algo in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algo]
        plt.scatter(algo_data['Runtime (s)'], algo_data['Residual Metric'],
                   label=algo, alpha=0.6)
    
    plt.title('Runtime vs Quality Trade-off by Algorithm')
    plt.xlabel('Runtime (s)')
    plt.ylabel('Residual Metric')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(comp_dir, 'runtime_quality_comparison.png'))
    plt.close()

def main():
    # Get project root and results directory
    project_root = get_project_root()
    results_dir = os.path.join(project_root, 'results')
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Read and process data
    print("Reading result files...")
    df = read_result_files()
    
    if df is None:
        return
    
    print(f"Processed {len(df)} result files")
    
    # Generate visualizations
    print("Generating visualizations...")
    generate_algorithm_specific_plots(df, results_dir)
    generate_comparison_plots(df, results_dir)
    
    # Save comprehensive CSV
    print("Saving comprehensive results...")
    df.to_csv(os.path.join(results_dir, 'comprehensive_results.csv'), index=False)
    
    print("\nAnalysis complete. Results saved to:")
    print("1. algorithm_analysis/ - Algorithm-specific visualizations")
    print("   - {algorithm}_search_area_effect.png")
    print("   - {algorithm}_block_size_effect.png")
    print("   - {algorithm}_runtime_quality.png")
    print("   - {algorithm}_heatmap.png")
    print("2. comparison_analysis/ - Cross-algorithm comparisons")
    print("   - search_area_comparison.png")
    print("   - block_size_comparison.png")
    print("   - runtime_quality_comparison.png")
    print("3. comprehensive_results.csv - Complete dataset")

if __name__ == "__main__":
    main() 