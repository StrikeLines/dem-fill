python run.py -p test -c config/dem_completion.json \
  --resume_state ./pretrained/100/100 \
  --n_timestep 100 \
  --input_img "/workspace/shared/dem-fill/test/11-2-25-b-test.tif" \
  --tile_overlap 12
