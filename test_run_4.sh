python run.py -p test -c config/dem_completion.json \
  --resume_state ./pretrained/100 \
  --n_timestep 512 \
  --input_img "/workspace/shared/proj-strikelines-ai-main/test/2-irregular.tif" \
  --input_mask "/workspace/shared/proj-strikelines-ai-main/test/2-irregular_nodata_mask.tif" \
  --tile_overlap 12
