#!/bin/bash

# Matrix Calculus EPUB Build Script
# This script converts the markdown file to EPUB format using pandoc

# Set script to exit on any error
set -e

# Configuration
INPUT_FILE="00-matrix-calc.md"
OUTPUT_FILE="output.epub"
TITLE="Matrix Calculus"
AUTHOR="Chris Snow"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    print_error "pandoc is not installed. Please install pandoc first."
    print_status "On macOS: brew install pandoc"
    print_status "On Ubuntu/Debian: sudo apt-get install pandoc"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    print_error "Input file '$INPUT_FILE' not found!"
    exit 1
fi

print_status "Starting EPUB build process..."
print_status "Input file: $INPUT_FILE"
print_status "Output file: $OUTPUT_FILE"
print_status "Title: $TITLE"
print_status "Author: $AUTHOR"

# Remove existing output file if it exists
if [ -f "$OUTPUT_FILE" ]; then
    print_warning "Removing existing output file: $OUTPUT_FILE"
    rm "$OUTPUT_FILE"
fi

# Build the EPUB using pandoc
print_status "Converting markdown to EPUB..."

pandoc "$INPUT_FILE" \
    -o "$OUTPUT_FILE" \
    --metadata title="$TITLE" \
    --metadata author="$AUTHOR" \
    --wrap=preserve

# Check if the build was successful
if [ -f "$OUTPUT_FILE" ]; then
    print_status "✅ EPUB build completed successfully!"
    print_status "Output file: $OUTPUT_FILE"
    
    # Get file size
    if command -v stat &> /dev/null; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            FILE_SIZE=$(stat -f%z "$OUTPUT_FILE")
        else
            # Linux
            FILE_SIZE=$(stat -c%s "$OUTPUT_FILE")
        fi
        print_status "File size: $(numfmt --to=iec-i --suffix=B $FILE_SIZE)"
    fi
    
    # Offer to open the file
    echo
    read -p "Would you like to open the EPUB file? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open "$OUTPUT_FILE"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            xdg-open "$OUTPUT_FILE"
        else
            print_status "Please open $OUTPUT_FILE manually with your preferred EPUB reader."
        fi
    fi
else
    print_error "❌ EPUB build failed!"
    exit 1
fi

print_status "Build script completed."
