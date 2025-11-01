# 🏔️ Diff-DEM: A Diffusion Probabilistic Approach to Digital Elevation Model Void Filling

## 🎯 Purpose

Diff-DEM is a deep learning-based tool designed to fill voids in Digital Elevation Models (DEMs) using a diffusion probabilistic model.  
It divides large georeferenced DEMs and their corresponding mask images into smaller tiles, processes them using a pretrained model, and stitches the results into full georeferenced output DEM.

---

## 📥 Input Format

To run inference, you need:

- ✅ An input DEM image (`--input_img`)
- ✅ A binary mask image (`--input_mask`) where voids are marked Or will be created from input image
- ✅ A pretrained model checkpoint (`--resume_state`)
- ✅ Number of diffusion inference steps (`--n_timestep`, e.g. 512)
- ✅ (Optional) Tile overlap in pixels (`--tile_overlap`) to reduce tiling artifacts

**Example Command:**

```bash
python run.py -p test -c config/dem_completion.json \
  --resume_state ./pretrained/760 \
  --n_timestep 512 \
  --input_img "Path_to_Input_DEM.tif" \
  --input_mask "Path_to_Mask.tif" \
  --tile_overlap 0
```

---

## 🧠 Internal Workflow

1. **Temporary Directory Creation**  
   A temp folder is created to hold intermediate tiles and results.

2. **Tiling**  
   - Input DEM and mask are split into 128×128 tiles.  
   - If `--tile_overlap` is provided (e.g., 64), tiles overlap with neighbors by that many pixels.  

3. **FList Creation**  
   Tile paths are written into `.flist` files and fed to the model.

4. **Model Inference**  
   - Model input remains **128×128**, regardless of overlap.  
   - With overlap, each pixel may be predicted multiple times (e.g., 50% overlap → predicted 3+ times).
   - Tiles values are converted into -1 to 1 ranges for model input/inference.
   - Tile's min_max values are saved using which the predicted images pixel values are scaled back to original 32bit dem pixel values.



5. **Output Stitching**  
   - Predicted tiles are merged back into a full DEM.  
   - Overlapping predictions are averaged, producing smooth transitions.  
   - Original georeferencing and metadata are preserved.

---

## 🔍 Tile Overlap: Why It Matters

Abrupt changes and seams may appear between tiles if no overlap is used. This happens because:  
- The model only “sees” one tile at a time.  
- Masks often span across tile boundaries.  
- No context beyond the current tile leads to discontinuities.  

| Option             | Description |
|--------------------|-------------|
| No `--tile_overlap` | Fast but may cause visible seams |
| With `--tile_overlap` | Slightly slower, but smoother and seamless outputs |

📌 Note: Overlap introduces some redundancy but improves quality.

---

## 🖥️ Environment Setup

```bash
conda env create -f environment.yml
conda activate Diff-DEM
```

---

## 📦 Dataset & Pretrained Model

- 🔽 Download: [Google Drive Link](https://drive.google.com/drive/folders/1RXlo2fl-TzGtA1WH5xE3TbzNWsHINln5?usp=sharing)
- 📁 Place dataset under: `Diff-DEM/dataset/norway_dem/` or give input images
- 📁 Place pretrained models at: `Diff-DEM/pretrained/`

---

## ⚙️ Command Line Arguments

```bash
python run.py -p test -c config/dem_completion.json \
  --resume_state ./pretrained/760 \
  --n_timestep 512 \
  --input_img "Input Image Path" \
  --input_mask "Input Mask Path" \
  --tile_overlap 20
```

| Argument | Description |
|----------|-------------|
| -p, --phase | train or test (default: test) |
| -c, --config | JSON config file |
| -b, --batch | Batch size (per GPU, under development) |
| --resume_state | Path to model checkpoint |
| --input_img | Input DEM (GeoTIFF) |
| --input_mask | Binary mask image |
| --tile_overlap | Overlap in pixels for tiling (default: 0) |
| --nodata_value | Value treated as NoData in input image for mask creation |
| --scale_factor | Rescale output (default: 1) |
| --keep_temp | Keep temporary tile outputs |
| --gpu_ids | GPU IDs to use (e.g., 0, 0,1) |
| --output_dir_name | Custom output directory name |
| --preprocess_type | Optional pre-processing flag |
| --data_root | Path to `.flist` of DEMs (batch mode) |
| --mask_root | Path to `.flist` of masks |
| --out_type | Output datatype (default: float32) |
| --use_color_map | Enable color map for TensorBoard |
| --sample_num | Number of samples to generate |
| --debug | Enable debug mode |
| --port | Visualization port (default: 21012) |

---

## 📈 Inference Modes

### Standard `.flist` Mode (Batch)

```bash
python run.py -p test -c config/dem_completion.json \
  --resume_state ./pretrained/760 \
  --n_timestep 512 \
  --data_root ./dataset/norway_dem/benchmark/benchmark_gt.flist \
  --mask_root ./dataset/norway_dem/benchmark/mask_64-96.flist
```

### Large Single Image Mode (With Tiling)

```bash
python run.py -p test -c config/dem_completion.json \
  --resume_state ./pretrained/760 \
  --n_timestep 512 \
  --input_img "Input_DEM.tif" \
  --input_mask "Input_Mask.tif" \
  --tile_overlap 64
```

---

## 🧪 Training

```bash
python run.py -p train -c config/dem_completion.json
```

Check training progress:

```bash
tensorboard --logdir experiments/train_dem_completion_XXXXXX_XXXXXX
```

---

## 🧪 Metric Evaluation

```bash
python data/util/tif_metric.py \
  --gt_tif_dir ./dataset/norway_dem/benchmark/gt \
  --mask_dir ./dataset/norway_dem/benchmark/mask/128-160 \
  --algo_dir ./experiments/Diff-DEM/128-160/results/test/0 \
  --normalize
```


---

## 📝 Known Issues

- **Tile artifacts**: Without `--tile_overlap`, visible seams may occur due to lack of tile context.  
- **Batch processing (`-b`)**: Batch option was implemented in source code but throws error therefore testing/development is underway.  
- **Model Input Size**: Always expects 128×128 inputs. Overlap is handled internally—model is agnostic to it.

---

## 🙏 Acknowledgements

- [Palette: Image-to-Image Diffusion Models](https://arxiv.org/abs/2111.05826)  
- [Gavriil's DEM Dataset](https://github.com/konstantg/dem-fill)  
- [Norwegian Mapping Authority](https://hoydedata.no/LaserInnsyn2/)  

---

## 📚 Citation

```bibtex
@article{lo2024diff,
  title={Diff-DEM: A Diffusion Probabilistic Approach to Digital Elevation Model Void Filling},
  author={Lo, Kyle Shih-Huang and Peters, Jörg},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2024},
  publisher={IEEE}
}
```