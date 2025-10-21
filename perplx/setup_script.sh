#!/bin/bash

# Nutrimood Chatbot Setup Script
# This script sets up the complete project structure and dependencies

set -e  # Exit on error

echo "=========================================="
echo "🍽️  Nutrimood Chatbot Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )(.+)')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python $REQUIRED_VERSION or higher is required"
    exit 1
fi

# Create project structure
echo ""
echo "Creating project structure..."

mkdir -p services
mkdir -p utils
mkdir -p tests
mkdir -p data
mkdir -p logs

print_success "Project directories created"

# Create __init__.py files
echo ""
echo "Creating initialization files..."

touch services/__init__.py
touch utils/__init__.py
touch tests/__init__.py

print_success "Initialization files created"

# Create virtual environment
echo ""
echo "Creating virtual environment..."

if [ -d "venv" ]; then
    print_warning "Virtual environment already exists"
else
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
print_success "Dependencies installed"

# Setup environment file
echo ""
echo "Setting up environment configuration..."

if [ -f ".env" ]; then
    print_warning ".env file already exists. Skipping..."
else
    cp .env.example .env
    print_success ".env file created from template"
    print_warning "Please edit .env with your AWS credentials"
fi

# Create sample food data if not exists
echo ""
echo "Checking food data..."

if [ -f "data/food_items.json" ]; then
    print_success "Food data already exists"
else
    print_warning "Food data not found. Please add your food_items.json to data/ directory"
fi

# Check AWS CLI
echo ""
echo "Checking AWS CLI..."

if command -v aws &> /dev/null; then
    print_success "AWS CLI found"
    
    # Test AWS credentials
    if aws sts get-caller-identity &> /dev/null; then
        print_success "AWS credentials configured"
    else
        print_warning "AWS credentials not configured. Please run 'aws configure'"
    fi
else
    print_warning "AWS CLI not found. Install it for easier AWS configuration"
fi

# Create logs directory structure
echo ""
echo "Setting up logging..."
touch logs/.gitkeep
print_success "Logging configured"

# Summary
echo ""
echo "=========================================="
echo "📋 Setup Summary"
echo "=========================================="
echo ""
echo "Project structure: ✓"
echo "Virtual environment: ✓"
echo "Dependencies: ✓"
echo "Configuration: ✓"
echo ""

# Next steps
echo "=========================================="
echo "🚀 Next Steps"
echo "=========================================="
echo ""
echo "1. Edit .env file with your AWS credentials"
echo "2. Add your food data to data/food_items.json"
echo "3. Activate virtual environment: source venv/bin/activate"
echo "4. Run the application: python main.py"
echo "5. Visit http://localhost:8000/docs for API documentation"
echo ""

# Optional: Run tests
read -p "Do you want to install dev dependencies and run tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install pytest pytest-cov
    print_success "Dev dependencies installed"
    
    if [ -f "tests/test_api.py" ]; then
        echo "Running tests..."
        pytest tests/ -v
    else
        print_warning "No tests found"
    fi
fi

echo ""
print_success "Setup complete! 🎉"
echo ""
