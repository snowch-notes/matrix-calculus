#!/bin/bash

# File: build.sh

# Ensure Quarto is installed
if ! command -v quarto &> /dev/null
then
    echo "Quarto CLI not found. Please install Quarto: https://quarto.org/docs/get-started/"
    exit 1
fi

# Define the Quarto file
QUARTO_FILE="matrix-calculus.qmd"

# Check if the Quarto file exists
if [ ! -f "$QUARTO_FILE" ]; then
    echo "Error: $QUARTO_FILE not found in the current directory."
    exit 1
fi

# Render to EPUB
echo "Rendering to EPUB..."
quarto render "$QUARTO_FILE" --to epub

