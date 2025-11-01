python run.py -p test -c config/dem_completion.json \
  --resume_state ./pretrained/100/100 \
  --n_timestep 512 \
  --input_img "./test/14-small-1.tif" \
  --input_mask "./test/14-small-1_nodata_mask.tif" \
  --tile_overlap 12
