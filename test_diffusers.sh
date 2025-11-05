#!/bin/bash

# Test script for diffusers integration
# This script demonstrates different inference methods and their speed improvements

echo "==================================================="
echo "🚀 Testing Diffusers Integration for Fast Inference"
echo "==================================================="

# Check if diffusers is installed
python -c "import diffusers; print('✅ Diffusers installation OK')" || {
    echo "❌ Diffusers not found. Installing..."
    pip install diffusers>=0.24.0 accelerate
}

# Test data - replace these paths with your actual test data
INPUT_DEM="test/11-2-25-test.tif"
INPUT_MASK="test/11-2-25-test_nodata_mask.tif"
MODEL_PATH="./pretrained/760"

echo -e "\n🔬 Running benchmark comparison..."

# Benchmark all methods (this will take some time but show the speedup)
echo "Running comprehensive benchmark (original vs all diffusers methods)..."
python run.py -p test -c config/dem_completion.json \
  --resume_state $MODEL_PATH \
  --input_img $INPUT_DEM \
  --input_mask $INPUT_MASK \
  --benchmark_inference \
  --tile_overlap 0 \
  --sample_num 0

echo -e "\n🏃‍♂️ Testing DPM-Solver++ (20 steps, ~25x speedup)..."
python run.py -p test -c config/dem_completion.json \
  --resume_state $MODEL_PATH \
  --input_img $INPUT_DEM \
  --input_mask $INPUT_MASK \
  --use_diffusers \
  --scheduler_type dpmpp \
  --inference_steps 20 \
  --tile_overlap 64 \
  --sample_num 0

echo -e "\n🔥 Testing UniPC (10 steps, ~50x speedup)..."
python run.py -p test -c config/dem_completion.json \
  --resume_state $MODEL_PATH \
  --input_img $INPUT_DEM \
  --input_mask $INPUT_MASK \
  --use_diffusers \
  --scheduler_type unipc \
  --inference_steps 10 \
  --tile_overlap 64 \
  --sample_num 0

echo -e "\n📋 Testing DDIM (50 steps, ~10x speedup)..."
python run.py -p test -c config/dem_completion.json \
  --resume_state $MODEL_PATH \
  --input_img $INPUT_DEM \
  --input_mask $INPUT_MASK \
  --use_diffusers \
  --scheduler_type ddim \
  --inference_steps 50 \
  --tile_overlap 64 \
  --sample_num 0

echo -e "\n✅ All tests completed!"
echo "Check the experiments/ folder for output results"
echo "Compare the timing in the logs to see the speedup"

echo -e "\n💡 Quick usage examples:"
echo "# Ultra-fast (10 steps): --use_diffusers --scheduler_type unipc --inference_steps 10"
echo "# Balanced (20 steps):   --use_diffusers --scheduler_type dpmpp --inference_steps 20" 
echo "# High quality (50):     --use_diffusers --scheduler_type ddim --inference_steps 50"
echo "# Original (512+ steps): (no diffusers flags)"