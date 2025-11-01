# Tile World File Generator

This utility generates `.twf` world files for tiled TIFF files to preserve georeferencing information, allowing tiles to be correctly projected on maps without waiting for the entire inference run to finish.

## Problem

When the DEM inpainting process tiles large georeferenced images, the individual tile files lose their georeferencing information. This means you can't properly visualize or analyze the tiles in GIS software until the entire inference is complete and the tiles are merged back together.

## Solution

The `generate_tile_worldfiles.py` script creates World Files (`.twf`) for each tile based on:
- The original image's georeferencing information
- The tile's position determined from its filename
- The tiling parameters (tile size, overlap)

## Files

- `generate_tile_worldfiles.py` - Main script for generating world files
- `example_worldfile_usage.py` - Example usage and verification script
- `README_worldfiles.md` - This documentation

## Usage

### Basic Usage

```bash
python generate_tile_worldfiles.py --input /path/to/original.tif --tiles_dir /path/to/tiles
```

### Full Usage

```bash
python generate_tile_worldfiles.py \
    --input /path/to/original_dem.tif \
    --tiles_dir /path/to/experiment/results/test/0 \
    --tile_size 128 \
    --overlap 0 \
    --prefix tile \
    --force
```

### Parameters

- `--input, -i`: Path to original georeferenced image (required)
- `--tiles_dir, -t`: Directory containing tile files (required)
- `--tile_size, -s`: Size of tiles in pixels (default: 128)
- `--overlap, -o`: Overlap between tiles in pixels (default: 0)
- `--prefix, -p`: Tile filename prefix (default: "tile")
- `--force, -f`: Overwrite existing world files

## How It Works

1. **Read Original Georeference**: Loads the original image's coordinate reference system (CRS) and affine transform
2. **Parse Tile Names**: Extracts row and column indices from tile filenames (e.g., `tile_0001_0002.tif`)
3. **Calculate Tile Position**: Determines each tile's geographic bounds based on its grid position
4. **Generate World File**: Creates a `.twf` file with the 6 world file parameters:
   - X-pixel size (west-east)
   - Rotation about Y-axis (typically 0)
   - Rotation about X-axis (typically 0)
   - Y-pixel size (north-south, negative)
   - X-coordinate of upper-left pixel center
   - Y-coordinate of upper-left pixel center

## Expected Filename Format

Tiles must follow the naming convention used by `GeoTiffTiler`:
```
{prefix}_{row:04d}_{col:04d}.tif
```

Examples:
- `tile_0000_0000.tif`
- `tile_0001_0002.tif`
- `mask_0003_0004.tif`

## Example Workflow

1. **Run DEM Inpainting**: Start your inference process
   ```bash
   python run.py -c config/dem_completion.json -p test -i /path/to/dem.tif
   ```

2. **Generate World Files**: While inference is running, generate world files for the tiles
   ```bash
   python generate_tile_worldfiles.py \
       --input /path/to/dem.tif \
       --tiles_dir experiments/your_experiment/results/test/0
   ```

3. **View in GIS**: Load the tiles with their world files in QGIS, ArcGIS, or other GIS software

## Verification

Use the example script to verify the world files:

```bash
python example_worldfile_usage.py
```

This will:
- Test filename parsing
- Generate example world files
- Verify the files can be read correctly
- Display georeferencing parameters

## Supported Formats

- **Input**: Any georeferenced raster format supported by GDAL/rasterio (GeoTIFF, etc.)
- **Output**: TIFF World Files (`.twf`) compatible with most GIS software
- **CRS**: Preserves the original image's coordinate reference system

## Integration with Existing Workflow

The script is designed to work with the existing `GeoTiffTiler` class in `core/util.py`. It uses the same tiling logic to calculate tile positions and transforms.

For automatic integration, you could modify the tiling process to generate world files immediately after creating tiles:

```python
# In your tiling code
from generate_tile_worldfiles import generate_worldfiles_for_tiles

# After tiling
generate_worldfiles_for_tiles(
    input_image_path=original_image,
    tiles_directory=tiles_output_dir,
    tile_size=128,
    overlap=0
)
```

## Troubleshooting

### Common Issues

1. **No tiles found**: Check that the `--tiles_dir` path is correct and contains `.tif` files
2. **Parsing errors**: Verify tiles follow the expected naming convention
3. **CRS issues**: Ensure the original image has valid georeferencing information

### Verification Steps

1. Check that world files are created (`.twf` extension)
2. Verify world file content has 6 numeric lines
3. Load tiles in GIS software to confirm proper positioning
4. Compare tile positions with original image extent

### Example World File Content

```
0.000833333333333
0.000000000000000
0.000000000000000
-0.000833333333333
-180.000416666666667
90.000416666666667
```

This represents:
- Pixel size: ~30 arc-seconds (common for DEM data)
- No rotation
- Geographic coordinates (latitude/longitude)