#!/bin/bash

#### For Slurm to run conda
# Script to run all TOML configuration files through runner.sh with separate log files
# Usage: ./run_all_configs.sh [--dry-run]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if dry-run mode
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}Running in dry-run mode - commands will be printed but not executed${NC}"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Find all TOML files in configs directory
echo -e "${BLUE}Finding all TOML configuration files...${NC}"
toml_files=$(find configs/ -name "*.toml" -type f | sort)

if [[ -z "$toml_files" ]]; then
    echo -e "${RED}No TOML files found in configs/ directory${NC}"
    exit 1
fi

# Count total files
total_files=$(echo "$toml_files" | wc -l)
echo -e "${GREEN}Found $total_files TOML configuration files${NC}"
echo

current=1
for config_file in $toml_files; do
    # Create log filename by replacing path separators with underscores
    # Remove configs/ prefix and .toml suffix, replace / with _
    log_name=$(echo "$config_file" | sed 's|^configs/||; s|\.toml$||; s|/|_|g')
    log_file="logs/${log_name}.log"
    
    echo -e "${BLUE}[$current/$total_files] Processing: ${config_file}${NC}"
    echo -e "${YELLOW}  Log file: ${log_file}${NC}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}  Would run: ./runner.sh \"$config_file\" &> \"$log_file\"${NC}"
    else
        # Run the configuration through runner.sh with output redirected to log file
        echo "Starting $(date)" > "$log_file"
        echo "Config: $config_file" >> "$log_file"
        echo "========================" >> "$log_file"
        
        if ./runner.sh "$config_file" &>> "$log_file"; then
            echo -e "${GREEN}  ✓ Completed successfully${NC}"
            echo "Completed successfully at $(date)" >> "$log_file"
        else
            echo -e "${RED}  ✗ Failed with exit code $?${NC}"
            echo "Failed with exit code $? at $(date)" >> "$log_file"
        fi
    fi
    
    echo
    current=$((current + 1))
done

if [[ "$DRY_RUN" == "false" ]]; then
    echo -e "${GREEN}All configurations processed!${NC}"
    echo -e "${BLUE}Log files are available in the logs/ directory${NC}"
    echo -e "${YELLOW}To view a specific log: tail -f logs/[config_name].log${NC}"
else
    echo -e "${YELLOW}Dry-run completed. Use './run_all_configs.sh' to actually run the configurations${NC}"
fi