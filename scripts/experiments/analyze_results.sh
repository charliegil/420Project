#!/bin/bash

# Create CSV file for all results
echo "Algorithm,Block Size,Search Area,Residual Metric,Naive Residual Metric,Runtime (s)" > results.csv

# Function to extract metrics from a result file
extract_metrics() {
    local file=$1
    local algorithm=$2
    local block_size=$3
    local search_area=$4
    
    if [ -f "$file" ]; then
        local residual_metric=$(grep "^Residual Metric:" "$file" | awk '{print $3}')
        local naive_residual_metric=$(grep "^Naive Residual Metric:" "$file" | awk '{print $4}')
        local runtime=$(grep "^Runtime:" "$file" | awk '{print $2}')
        
        echo "$algorithm,$block_size,$search_area,$residual_metric,$naive_residual_metric,$runtime" >> results.csv
        return 0
    else
        return 1
    fi
}

# Process all result files
for file in results/*.txt; do
    if [[ "$file" == *"summary.txt"* ]]; then
        continue
    fi
    
    # Extract algorithm, block size, and search area from filename
    filename=$(basename "$file")
    
    if [[ "$filename" == "full_block"* ]]; then
        algorithm="Full Search"
        block_size=$(echo "$filename" | grep -o "block[0-9]*" | grep -o "[0-9]*")
        search_area=$(echo "$filename" | grep -o "search[0-9]*" | grep -o "[0-9]*")
    elif [[ "$filename" == "diamond_block"* ]]; then
        algorithm="Diamond Search"
        block_size=$(echo "$filename" | grep -o "block[0-9]*" | grep -o "[0-9]*")
        search_area=$(echo "$filename" | grep -o "search[0-9]*" | grep -o "[0-9]*")
    elif [[ "$filename" == "three_step_block"* ]]; then
        algorithm="Three Step Search"
        block_size=$(echo "$filename" | grep -o "block[0-9]*" | grep -o "[0-9]*")
        search_area=$(echo "$filename" | grep -o "search[0-9]*" | grep -o "[0-9]*")
    fi
    
    extract_metrics "$file" "$algorithm" "$block_size" "$search_area"
done

# Generate analysis report
echo "=== MOTION ESTIMATION ALGORITHM ANALYSIS ===" > analysis_report.txt
echo "" >> analysis_report.txt

# Calculate average metrics for each algorithm
three_step_count=0
three_step_avg_residual=0
three_step_avg_runtime=0

full_search_count=0
full_search_avg_residual=0
full_search_avg_runtime=0

diamond_search_count=0
diamond_search_avg_residual=0
diamond_search_avg_runtime=0

while IFS=, read -r algorithm block_size search_area residual naive runtime; do
    if [[ "$algorithm" == "Three Step Search" ]]; then
        three_step_count=$((three_step_count + 1))
        three_step_avg_residual=$(echo "$three_step_avg_residual + $residual" | bc)
        three_step_avg_runtime=$(echo "$three_step_avg_runtime + $runtime" | bc)
    elif [[ "$algorithm" == "Full Search" ]]; then
        full_search_count=$((full_search_count + 1))
        full_search_avg_residual=$(echo "$full_search_avg_residual + $residual" | bc)
        full_search_avg_runtime=$(echo "$full_search_avg_runtime + $runtime" | bc)
    elif [[ "$algorithm" == "Diamond Search" ]]; then
        diamond_search_count=$((diamond_search_count + 1))
        diamond_search_avg_residual=$(echo "$diamond_search_avg_residual + $residual" | bc)
        diamond_search_avg_runtime=$(echo "$diamond_search_avg_runtime + $runtime" | bc)
    fi
done < results.csv

# Calculate averages
if [ "$three_step_count" -gt 0 ]; then
    three_step_avg_residual=$(echo "scale=6; $three_step_avg_residual / $three_step_count" | bc)
    three_step_avg_runtime=$(echo "scale=6; $three_step_avg_runtime / $three_step_count" | bc)
fi

if [ "$full_search_count" -gt 0 ]; then
    full_search_avg_residual=$(echo "scale=6; $full_search_avg_residual / $full_search_count" | bc)
    full_search_avg_runtime=$(echo "scale=6; $full_search_avg_runtime / $full_search_count" | bc)
fi

if [ "$diamond_search_count" -gt 0 ]; then
    diamond_search_avg_residual=$(echo "scale=6; $diamond_search_avg_residual / $diamond_search_count" | bc)
    diamond_search_avg_runtime=$(echo "scale=6; $diamond_search_avg_runtime / $diamond_search_count" | bc)
fi

# Print overall algorithm comparison
echo "=== OVERALL ALGORITHM COMPARISON ===" >> analysis_report.txt
echo "" >> analysis_report.txt
echo "Three Step Search:" >> analysis_report.txt
echo "  Average Residual Metric: $three_step_avg_residual" >> analysis_report.txt
echo "  Average Runtime: $three_step_avg_runtime seconds" >> analysis_report.txt
echo "" >> analysis_report.txt
echo "Full Search:" >> analysis_report.txt
echo "  Average Residual Metric: $full_search_avg_residual" >> analysis_report.txt
echo "  Average Runtime: $full_search_avg_runtime seconds" >> analysis_report.txt
echo "" >> analysis_report.txt
echo "Diamond Search:" >> analysis_report.txt
echo "  Average Residual Metric: $diamond_search_avg_residual" >> analysis_report.txt
echo "  Average Runtime: $diamond_search_avg_runtime seconds" >> analysis_report.txt
echo "" >> analysis_report.txt

# Calculate speedups
if [ "$three_step_count" -gt 0 ] && [ "$full_search_count" -gt 0 ]; then
    speedup_full_vs_three=$(echo "scale=2; $full_search_avg_runtime / $three_step_avg_runtime" | bc)
    echo "Three Step Search is ${speedup_full_vs_three}x faster than Full Search" >> analysis_report.txt
fi

if [ "$diamond_search_count" -gt 0 ] && [ "$full_search_count" -gt 0 ]; then
    speedup_full_vs_diamond=$(echo "scale=2; $full_search_avg_runtime / $diamond_search_avg_runtime" | bc)
    echo "Diamond Search is ${speedup_full_vs_diamond}x faster than Full Search" >> analysis_report.txt
fi

if [ "$diamond_search_count" -gt 0 ] && [ "$three_step_count" -gt 0 ]; then
    speedup_three_vs_diamond=$(echo "scale=2; $three_step_avg_runtime / $diamond_search_avg_runtime" | bc)
    echo "Diamond Search is ${speedup_three_vs_diamond}x faster than Three Step Search" >> analysis_report.txt
fi

echo "" >> analysis_report.txt

# Generate block size comparison
echo "=== BLOCK SIZE COMPARISON ===" >> analysis_report.txt
echo "" >> analysis_report.txt
echo "Three Step Search Algorithm:" >> analysis_report.txt
echo "Block Size | Residual Metric | Runtime (s) | Speedup vs Full Search" >> analysis_report.txt
echo "-----------|----------------|-------------|----------------------" >> analysis_report.txt

for block_size in 4 8 16 32; do
    three_step_file="results/three_step_block${block_size}_search7.txt"
    full_search_file="results/full_block${block_size}_search7.txt"
    diamond_search_file="results/diamond_block${block_size}_search7.txt"
    
    if [ -f "$three_step_file" ] && [ -f "$full_search_file" ]; then
        three_step_residual=$(grep "^Residual Metric:" "$three_step_file" | awk '{print $3}')
        three_step_runtime=$(grep "^Runtime:" "$three_step_file" | awk '{print $2}')
        full_search_runtime=$(grep "^Runtime:" "$full_search_file" | awk '{print $2}')
        
        # Calculate speedup
        speedup=$(echo "scale=2; $full_search_runtime / $three_step_runtime" | bc)
        
        echo "$block_size | $three_step_residual | $three_step_runtime | ${speedup}x" >> analysis_report.txt
    fi
done

echo "" >> analysis_report.txt
echo "Full Search Algorithm:" >> analysis_report.txt
echo "Block Size | Residual Metric | Runtime (s)" >> analysis_report.txt
echo "-----------|----------------|-------------" >> analysis_report.txt

for block_size in 4 8 16 32; do
    full_search_file="results/full_block${block_size}_search7.txt"
    
    if [ -f "$full_search_file" ]; then
        full_search_residual=$(grep "^Residual Metric:" "$full_search_file" | awk '{print $3}')
        full_search_runtime=$(grep "^Runtime:" "$full_search_file" | awk '{print $2}')
        
        echo "$block_size | $full_search_residual | $full_search_runtime" >> analysis_report.txt
    fi
done

echo "" >> analysis_report.txt
echo "Diamond Search Algorithm:" >> analysis_report.txt
echo "Block Size | Residual Metric | Runtime (s) | Speedup vs Full Search" >> analysis_report.txt
echo "-----------|----------------|-------------|----------------------" >> analysis_report.txt

for block_size in 4 8 16 32; do
    diamond_search_file="results/diamond_block${block_size}_search7.txt"
    full_search_file="results/full_block${block_size}_search7.txt"
    
    if [ -f "$diamond_search_file" ] && [ -f "$full_search_file" ]; then
        diamond_search_residual=$(grep "^Residual Metric:" "$diamond_search_file" | awk '{print $3}')
        diamond_search_runtime=$(grep "^Runtime:" "$diamond_search_file" | awk '{print $2}')
        full_search_runtime=$(grep "^Runtime:" "$full_search_file" | awk '{print $2}')
        
        # Calculate speedup
        speedup=$(echo "scale=2; $full_search_runtime / $diamond_search_runtime" | bc)
        
        echo "$block_size | $diamond_search_residual | $diamond_search_runtime | ${speedup}x" >> analysis_report.txt
    fi
done

# Generate search area comparison
echo "" >> analysis_report.txt
echo "=== SEARCH AREA COMPARISON ===" >> analysis_report.txt
echo "" >> analysis_report.txt
echo "Three Step Search Algorithm:" >> analysis_report.txt
echo "Search Area | Residual Metric | Runtime (s) | Speedup vs Full Search" >> analysis_report.txt
echo "------------|----------------|-------------|----------------------" >> analysis_report.txt

for search_area in 3 5 7 9 11; do
    three_step_file="results/three_step_block16_search${search_area}.txt"
    full_search_file="results/full_block16_search${search_area}.txt"
    
    if [ -f "$three_step_file" ] && [ -f "$full_search_file" ]; then
        three_step_residual=$(grep "^Residual Metric:" "$three_step_file" | awk '{print $3}')
        three_step_runtime=$(grep "^Runtime:" "$three_step_file" | awk '{print $2}')
        full_search_runtime=$(grep "^Runtime:" "$full_search_file" | awk '{print $2}')
        
        # Calculate speedup
        speedup=$(echo "scale=2; $full_search_runtime / $three_step_runtime" | bc)
        
        echo "$search_area | $three_step_residual | $three_step_runtime | ${speedup}x" >> analysis_report.txt
    fi
done

echo "" >> analysis_report.txt
echo "Full Search Algorithm:" >> analysis_report.txt
echo "Search Area | Residual Metric | Runtime (s)" >> analysis_report.txt
echo "------------|----------------|-------------" >> analysis_report.txt

for search_area in 3 5 7 9 11; do
    full_search_file="results/full_block16_search${search_area}.txt"
    
    if [ -f "$full_search_file" ]; then
        full_search_residual=$(grep "^Residual Metric:" "$full_search_file" | awk '{print $3}')
        full_search_runtime=$(grep "^Runtime:" "$full_search_file" | awk '{print $2}')
        
        echo "$search_area | $full_search_residual | $full_search_runtime" >> analysis_report.txt
    fi
done

echo "" >> analysis_report.txt
echo "Diamond Search Algorithm:" >> analysis_report.txt
echo "Search Area | Residual Metric | Runtime (s) | Speedup vs Full Search" >> analysis_report.txt
echo "------------|----------------|-------------|----------------------" >> analysis_report.txt

for search_area in 3 5 7 9 11; do
    diamond_search_file="results/diamond_block16_search${search_area}.txt"
    full_search_file="results/full_block16_search${search_area}.txt"
    
    if [ -f "$diamond_search_file" ] && [ -f "$full_search_file" ]; then
        diamond_search_residual=$(grep "^Residual Metric:" "$diamond_search_file" | awk '{print $3}')
        diamond_search_runtime=$(grep "^Runtime:" "$diamond_search_file" | awk '{print $2}')
        full_search_runtime=$(grep "^Runtime:" "$full_search_file" | awk '{print $2}')
        
        # Calculate speedup
        speedup=$(echo "scale=2; $full_search_runtime / $diamond_search_runtime" | bc)
        
        echo "$search_area | $diamond_search_residual | $diamond_search_runtime | ${speedup}x" >> analysis_report.txt
    fi
done

# Generate combined parameters comparison
echo "" >> analysis_report.txt
echo "=== COMBINED PARAMETERS COMPARISON ===" >> analysis_report.txt
echo "" >> analysis_report.txt
echo "Three Step Search Algorithm:" >> analysis_report.txt
echo "Block Size | Search Area | Residual Metric | Runtime (s) | Speedup vs Full Search" >> analysis_report.txt
echo "-----------|------------|----------------|-------------|----------------------" >> analysis_report.txt

for block_size in 8 32; do
    for search_area in 3 5 9; do
        three_step_file="results/three_step_block${block_size}_search${search_area}.txt"
        full_search_file="results/full_block${block_size}_search${search_area}.txt"
        
        if [ -f "$three_step_file" ] && [ -f "$full_search_file" ]; then
            three_step_residual=$(grep "^Residual Metric:" "$three_step_file" | awk '{print $3}')
            three_step_runtime=$(grep "^Runtime:" "$three_step_file" | awk '{print $2}')
            full_search_runtime=$(grep "^Runtime:" "$full_search_file" | awk '{print $2}')
            
            # Calculate speedup
            speedup=$(echo "scale=2; $full_search_runtime / $three_step_runtime" | bc)
            
            echo "$block_size | $search_area | $three_step_residual | $three_step_runtime | ${speedup}x" >> analysis_report.txt
        fi
    done
done

echo "" >> analysis_report.txt
echo "Full Search Algorithm:" >> analysis_report.txt
echo "Block Size | Search Area | Residual Metric | Runtime (s)" >> analysis_report.txt
echo "-----------|------------|----------------|-------------" >> analysis_report.txt

for block_size in 8 32; do
    for search_area in 3 5 9; do
        full_search_file="results/full_block${block_size}_search${search_area}.txt"
        
        if [ -f "$full_search_file" ]; then
            full_search_residual=$(grep "^Residual Metric:" "$full_search_file" | awk '{print $3}')
            full_search_runtime=$(grep "^Runtime:" "$full_search_file" | awk '{print $2}')
            
            echo "$block_size | $search_area | $full_search_residual | $full_search_runtime" >> analysis_report.txt
        fi
    done
done

echo "" >> analysis_report.txt
echo "Diamond Search Algorithm:" >> analysis_report.txt
echo "Block Size | Search Area | Residual Metric | Runtime (s) | Speedup vs Full Search" >> analysis_report.txt
echo "-----------|------------|----------------|-------------|----------------------" >> analysis_report.txt

for block_size in 8 32; do
    for search_area in 3 5 9; do
        diamond_search_file="results/diamond_block${block_size}_search${search_area}.txt"
        full_search_file="results/full_block${block_size}_search${search_area}.txt"
        
        if [ -f "$diamond_search_file" ] && [ -f "$full_search_file" ]; then
            diamond_search_residual=$(grep "^Residual Metric:" "$diamond_search_file" | awk '{print $3}')
            diamond_search_runtime=$(grep "^Runtime:" "$diamond_search_file" | awk '{print $2}')
            full_search_runtime=$(grep "^Runtime:" "$full_search_file" | awk '{print $2}')
            
            # Calculate speedup
            speedup=$(echo "scale=2; $full_search_runtime / $diamond_search_runtime" | bc)
            
            echo "$block_size | $search_area | $diamond_search_residual | $diamond_search_runtime | ${speedup}x" >> analysis_report.txt
        fi
    done
done

# Find best configurations
echo "" >> analysis_report.txt
echo "=== CONCLUSIONS ===" >> analysis_report.txt
echo "" >> analysis_report.txt

# Find best Three Step Search configuration
best_three_step_residual=999999
best_three_step_runtime=999999
best_three_step_block_size=0
best_three_step_search_area=0

while IFS=, read -r algorithm block_size search_area residual naive runtime; do
    if [[ "$algorithm" == "Three Step Search" ]]; then
        if (( $(echo "$residual < $best_three_step_residual" | bc -l) )); then
            best_three_step_residual=$residual
            best_three_step_runtime=$runtime
            best_three_step_block_size=$block_size
            best_three_step_search_area=$search_area
        fi
    fi
done < results.csv

echo "Best Three Step Search Configuration:" >> analysis_report.txt
echo "  Block Size: $best_three_step_block_size" >> analysis_report.txt
echo "  Search Area: size: $best_three_step_search_area" >> analysis_report.txt
echo "  Residual Metric: $best_three_step_residual" >> analysis_report.txt
echo "  Runtime: $best_three_step_runtime seconds" >> analysis_report.txt
echo "" >> analysis_report.txt

# Find best Full Search configuration
best_full_search_residual=999999
best_full_search_runtime=999999
best_full_search_block_size=0
best_full_search_search_area=0

while IFS=, read -r algorithm block_size search_area residual naive runtime; do
    if [[ "$algorithm" == "Full Search" ]]; then
        if (( $(echo "$residual < $best_full_search_residual" | bc -l) )); then
            best_full_search_residual=$residual
            best_full_search_runtime=$runtime
            best_full_search_block_size=$block_size
            best_full_search_search_area=$search_area
        fi
    fi
done < results.csv

echo "Best Full Search Configuration:" >> analysis_report.txt
echo "  Block Size: $best_full_search_block_size" >> analysis_report.txt
echo "  Search Area: size: $best_full_search_search_area" >> analysis_report.txt
echo "  Residual Metric: $best_full_search_residual" >> analysis_report.txt
echo "  Runtime: $best_full_search_runtime seconds" >> analysis_report.txt
echo "" >> analysis_report.txt

# Find best Diamond Search configuration
best_diamond_search_residual=999999
best_diamond_search_runtime=999999
best_diamond_search_block_size=0
best_diamond_search_search_area=0

while IFS=, read -r algorithm block_size search_area residual naive runtime; do
    if [[ "$algorithm" == "Diamond Search" ]]; then
        if (( $(echo "$residual < $best_diamond_search_residual" | bc -l) )); then
            best_diamond_search_residual=$residual
            best_diamond_search_runtime=$runtime
            best_diamond_search_block_size=$block_size
            best_diamond_search_search_area=$search_area
        fi
    fi
done < results.csv

echo "Best Diamond Search Configuration:" >> analysis_report.txt
echo "  Block Size: $best_diamond_search_block_size" >> analysis_report.txt
echo "  Search Area: size: $best_diamond_search_search_area" >> analysis_report.txt
echo "  Residual Metric: $best_diamond_search_residual" >> analysis_report.txt
echo "  Runtime: $best_diamond_search_runtime seconds" >> analysis_report.txt
echo "" >> analysis_report.txt

# Calculate overall speedups
if [ "$three_step_count" -gt 0 ] && [ "$full_search_count" -gt 0 ]; then
    speedup_full_vs_three=$(echo "scale=2; $full_search_avg_runtime / $three_step_avg_runtime" | bc)
    echo "Overall Speedup: Three Step Search is ${speedup_full_vs_three}x faster than Full Search" >> analysis_report.txt
fi

if [ "$diamond_search_count" -gt 0 ] && [ "$full_search_count" -gt 0 ]; then
    speedup_full_vs_diamond=$(echo "scale=2; $full_search_avg_runtime / $diamond_search_avg_runtime" | bc)
    echo "Overall Speedup: Diamond Search is ${speedup_full_vs_diamond}x faster than Full Search" >> analysis_report.txt
fi

if [ "$diamond_search_count" -gt 0 ] && [ "$three_step_count" -gt 0 ]; then
    speedup_three_vs_diamond=$(echo "scale=2; $three_step_avg_runtime / $diamond_search_avg_runtime" | bc)
    echo "Overall Speedup: Diamond Search is ${speedup_three_vs_diamond}x faster than Three Step Search" >> analysis_report.txt
fi

echo "" >> analysis_report.txt

# Add recommendations
echo "=== RECOMMENDATIONS ===" >> analysis_report.txt
echo "" >> analysis_report.txt
echo "1. For real-time applications:" >> analysis_report.txt
echo "   Use Diamond Search with block size $best_diamond_search_block_size and search area size: $best_diamond_search_search_area" >> analysis_report.txt
echo "2. For high-quality applications:" >> analysis_report.txt
echo "   Use Full Search with block size $best_full_search_block_size and search area size: $best_full_search_search_area" >> analysis_report.txt
echo "3. For a good balance between speed and quality:" >> analysis_report.txt
echo "   Use Diamond Search with block size $best_diamond_search_block_size and search area size: $best_diamond_search_search_area" >> analysis_report.txt

echo "Analysis complete. Results saved to analysis_report.txt and results.csv" 