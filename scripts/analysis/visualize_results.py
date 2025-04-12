#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Check if results.csv exists
if not os.path.exists('results.csv'):
    print("Error: results.csv not found. Please run analyze_results.sh first.")
    exit(1)

# Read the CSV file
df = pd.read_csv('results.csv')

# Create a directory for plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Create performance vs quality trade-off plot
plt.figure(figsize=(15, 10))
for algo in df['Algorithm'].unique():
    algo_data = df[df['Algorithm'] == algo]
    # Map algorithm names to shorter versions for the legend
    algo_name = algo.lower().replace(' search', '')
    # Scale the sizes to be more visible
    sizes = algo_data['Block Size'].map(lambda x: x * 30)
    plt.scatter(algo_data['Runtime (s)'], algo_data['Residual Metric'],
               s=sizes, alpha=0.7, label=algo_name)

plt.xscale('log')  # Use log scale for runtime
plt.xlabel('Runtime (s)', fontsize=12)
plt.ylabel('Residual Metric', fontsize=12)
plt.title('Performance vs Quality Trade-off', fontsize=14, pad=20)

# Add legend for algorithms
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add size legend
legend_elements = [plt.scatter([], [], s=size*30, c='gray', alpha=0.3, label=str(size))
                  for size in sorted(df['Block Size'].unique())]
second_legend = plt.legend(handles=legend_elements, title='Block Size',
                          bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.gca().add_artist(second_legend)  # Add the second legend

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/performance_quality_tradeoff.png', bbox_inches='tight', dpi=300)
plt.close()

# Create a figure with multiple subplots
plt.figure(figsize=(15, 10))

# 1. Block Size Comparison
plt.subplot(2, 2, 1)
for algo in df['Algorithm'].unique():
    algo_data = df[df['Algorithm'] == algo]
    plt.plot(algo_data['Block Size'], algo_data['Residual Metric'], marker='o', label=algo)
plt.xlabel('Block Size')
plt.ylabel('Residual Metric')
plt.title('Residual Metric vs Block Size')
plt.legend()
plt.grid(True)

# 2. Runtime vs Block Size
plt.subplot(2, 2, 2)
for algo in df['Algorithm'].unique():
    algo_data = df[df['Algorithm'] == algo]
    plt.plot(algo_data['Block Size'], algo_data['Runtime (s)'], marker='o', label=algo)
plt.xlabel('Block Size')
plt.ylabel('Runtime (s)')
plt.title('Runtime vs Block Size')
plt.legend()
plt.grid(True)

# 3. Search Area Comparison
plt.subplot(2, 2, 3)
for algo in df['Algorithm'].unique():
    algo_data = df[df['Algorithm'] == algo]
    plt.plot(algo_data['Search Area'], algo_data['Residual Metric'], marker='o', label=algo)
plt.xlabel('Search Area')
plt.ylabel('Residual Metric')
plt.title('Residual Metric vs Search Area')
plt.legend()
plt.grid(True)

# 4. Runtime vs Search Area
plt.subplot(2, 2, 4)
for algo in df['Algorithm'].unique():
    algo_data = df[df['Algorithm'] == algo]
    plt.plot(algo_data['Search Area'], algo_data['Runtime (s)'], marker='o', label=algo)
plt.xlabel('Search Area')
plt.ylabel('Runtime (s)')
plt.title('Runtime vs Search Area')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('motion_estimation_results.png')
print("Visualization saved as motion_estimation_results.png")

# Create a second figure for speedup analysis
plt.figure(figsize=(12, 8))

# Calculate speedup for each configuration
three_step_data = df[df['Algorithm'] == 'Three Step Search']
full_search_data = df[df['Algorithm'] == 'Full Search']

# Merge the dataframes on Block Size and Search Area
merged_data = pd.merge(three_step_data, full_search_data, 
                      on=['Block Size', 'Search Area'], 
                      suffixes=('_three_step', '_full_search'))

# Calculate speedup
merged_data['Speedup'] = merged_data['Runtime (s)_full_search'] / merged_data['Runtime (s)_three_step']

# Plot speedup vs block size
plt.subplot(1, 2, 1)
for search_area in merged_data['Search Area'].unique():
    area_data = merged_data[merged_data['Search Area'] == search_area]
    plt.plot(area_data['Block Size'], area_data['Speedup'], marker='o', label=f'Search Area: {search_area}')
plt.xlabel('Block Size')
plt.ylabel('Speedup (Full Search / Three Step Search)')
plt.title('Speedup vs Block Size')
plt.legend()
plt.grid(True)

# Plot speedup vs search area
plt.subplot(1, 2, 2)
for block_size in merged_data['Block Size'].unique():
    size_data = merged_data[merged_data['Block Size'] == block_size]
    plt.plot(size_data['Search Area'], size_data['Speedup'], marker='o', label=f'Block Size: {block_size}')
plt.xlabel('Search Area')
plt.ylabel('Speedup (Full Search / Three Step Search)')
plt.title('Speedup vs Search Area')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('speedup_analysis.png')
print("Speedup analysis saved as speedup_analysis.png")

# Create a third figure for quality comparison
plt.figure(figsize=(12, 8))

# Calculate quality difference
merged_data['Quality_Diff'] = merged_data['Residual Metric_three_step'] - merged_data['Residual Metric_full_search']
merged_data['Quality_Diff_Percent'] = (merged_data['Quality_Diff'] / merged_data['Residual Metric_full_search']) * 100

# Plot quality difference vs block size
plt.subplot(1, 2, 1)
for search_area in merged_data['Search Area'].unique():
    area_data = merged_data[merged_data['Search Area'] == search_area]
    plt.plot(area_data['Block Size'], area_data['Quality_Diff_Percent'], marker='o', label=f'Search Area: {search_area}')
plt.xlabel('Block Size')
plt.ylabel('Quality Difference (%)')
plt.title('Quality Difference vs Block Size')
plt.legend()
plt.grid(True)

# Plot quality difference vs search area
plt.subplot(1, 2, 2)
for block_size in merged_data['Block Size'].unique():
    size_data = merged_data[merged_data['Block Size'] == block_size]
    plt.plot(size_data['Search Area'], size_data['Quality_Diff_Percent'], marker='o', label=f'Block Size: {block_size}')
plt.xlabel('Search Area')
plt.ylabel('Quality Difference (%)')
plt.title('Quality Difference vs Search Area')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('quality_comparison.png')
print("Quality comparison saved as quality_comparison.png")

# Print summary statistics
print("\nSummary Statistics:")
print(f"Average Speedup: {merged_data['Speedup'].mean():.2f}x")
print(f"Average Quality Difference: {merged_data['Quality_Diff_Percent'].mean():.2f}%")
print(f"Best Configuration (Three Step Search):")
best_three_step = three_step_data.loc[three_step_data['Residual Metric'].idxmin()]
print(f"  Block Size: {best_three_step['Block Size']}")
print(f"  Search Area: {best_three_step['Search Area']}")
print(f"  Residual Metric: {best_three_step['Residual Metric']:.6f}")
print(f"  Runtime: {best_three_step['Runtime (s)']:.6f} seconds")
print(f"Best Configuration (Full Search):")
best_full_search = full_search_data.loc[full_search_data['Residual Metric'].idxmin()]
print(f"  Block Size: {best_full_search['Block Size']}")
print(f"  Search Area: {best_full_search['Search Area']}")
print(f"  Residual Metric: {best_full_search['Residual Metric']:.6f}")
print(f"  Runtime: {best_full_search['Runtime (s)']:.6f} seconds") 