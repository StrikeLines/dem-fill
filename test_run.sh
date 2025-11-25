# Example 1: With provided mask (original behavior)
python run.py -p test -c config/dem_completion.json \
  --resume_state pretrained/100 \
  --n_timestep 500 \
  --input_img "test/14-small-1.tif" \
  --tile_overlap 12 \
  --gpu_ids 0 \
  --batch 12

# Example 2: With auto-generated mask (new behavior - mask optional)
# python run.py -p test -c config/dem_completion.json \
#   --resume_state ./pretrained/100/100 \
#   --n_timestep 512 \
#   --input_img "./test/21-small.tif" \
#   --tile_overlap 12
