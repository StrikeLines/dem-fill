# Example 1: With provided mask (original behavior)
python run.py -p test -c config/dem_completion.json \
  --resume_state ./pretrained/100/100 \
  --n_timestep 512 \
  --input_img "/workspace/shared/dem-fill/test/ridge-test.tif" \
  --tile_overlap 64

# Example 2: With auto-generated mask (new behavior - mask optional)
# python run.py -p test -c config/dem_completion.json \
#   --resume_state ./pretrained/100/100 \
#   --n_timestep 512 \
#   --input_img "./test/21-small.tif" \
#   --tile_overlap 12
