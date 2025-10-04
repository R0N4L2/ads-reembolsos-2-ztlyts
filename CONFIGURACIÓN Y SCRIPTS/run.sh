#!/bin/bash

# FIFA World Cup 2018 Predictor - Run Script
# This script sets up the environment and runs the prediction model

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}FIFA World Cup 2018 Predictor${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python 3 found: $(python3 --version)"

# Check if data files exist
if [ ! -f "data/matches.csv" ]; then
    echo -e "${RED}Error: data/matches.csv not found${NC}"
    echo "Please ensure all data files are in the data/ directory"
    exit 1
fi

if [ ! -f "data/teams.csv" ]; then
    echo -e "${RED}Error: data/teams.csv not found${NC}"
    exit 1
fi

if [ ! -f "data/qualified.csv" ]; then
    echo -e "${RED}Error: data/qualified.csv not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} All data files found"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Install/upgrade dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✓${NC} Dependencies installed"

# Run the main script
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting prediction model...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

python main.py

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Simulation completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Results have been displayed above."
    echo "For detailed analysis, check RESULTS.md"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Simulation failed${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

# Deactivate virtual environment
deactivate
