#!/bin/bash

# RunPod Initialization Script for dem-fill project
set -e  # Exit on error

echo "Starting RunPod initialization..."

# Navigate to shared workspace
echo "Navigating to /workspace/shared..."
cd /workspace/shared

# Clone the repository with authentication
echo "Cloning dem-fill repository..."
GITHUB_TOKEN="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"
git clone https://${GITHUB_TOKEN}@github.com/StrikeLines/dem-fill.git

# Navigate into the cloned repository
echo "Entering dem-fill directory..."
cd /workspace/shared/dem-fill

# Set up password SSH access
echo "Setting up password SSH access..."
wget https://raw.githubusercontent.com/justinwlin/Runpod-SSH-Password/main/passwordrunpod.sh
chmod +x passwordrunpod.sh

# Run the SSH password setup script with password input
echo "Configuring SSH password..."
echo "400KhzSonar!" | ./passwordrunpod.sh

# Download the model from Google Drive
echo "Downloading model from Google Drive..."
wget --no-check-certificate 'https://drive.usercontent.google.com/download?id=1DFDqbCDj2eG-jWkop_GRIniuJs8YGnJ3&export=download&confirm=t&uuid=1c4eb783-6126-4132-9796-f35705b4d482' -O 100_dem_fill.tar

# Create pretrained directory if it doesn't exist
echo "Creating pretrained directory..."
mkdir -p /workspace/shared/dem-fill/pretrained

# Extract the model to the pretrained folder
echo "Extracting model to pretrained folder..."
tar -xf 100_dem_fill.tar -C /workspace/shared/dem-fill/pretrained

# Install Python requirements
echo "Installing Python requirements..."
pip install -r requirements.txt

echo "RunPod initialization complete!"
echo "Repository location: /workspace/shared/dem-fill"
