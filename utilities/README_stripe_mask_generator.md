# Stripe Pattern Mask Generator

This script generates stripe patterned mask files for training neural networks, specifically designed for DEM inpainting and similar tasks where simulated missing data patterns are needed.

## Features

- **Parallel stripe patterns** with random rotation (0-360°)
- **Variable stripe widths**:
  - Masked stripes: 20-90 pixels
  - Unmasked stripes: 40-100 pixels
- **Multiple tile sizes**: 128px, 256px, or 512px
- **TIF format output** with binary masks (0=valid, 255=masked)
- **LZW compression** for efficient storage
- **Random seed support** for reproducible generation

## Installation Requirements

The script requires the following Python packages:
```bash
pip install numpy pillow rasterio
```

## Usage

### Basic Usage
```bash
# Generate 100 masks of 128x128 pixels
python generate_stripe_masks.py --output_dir ./masks --tile_size 128 --num_tiles 100

# Generate 50 masks of 256x256 pixels with custom prefix
python generate_stripe_masks.py --output_dir ./training_masks --tile_size 256 --num_tiles 50 --prefix "stripe_pattern"

# Generate large 512x512 masks
python generate_stripe_masks.py --output_dir ./large_masks --tile_size 512 --num_tiles 25
```

### Command Line Arguments

- `--output_dir`, `-o` (required): Output directory for generated mask files
- `--tile_size`, `-s`: Square tile size in pixels (choices: 128, 256, 512, default: 128)
- `--num_tiles`, `-n` (required): Number of mask tiles to generate
- `--prefix`, `-p`: Filename prefix for generated masks (default: "stripe_mask")
- `--seed`: Random seed for reproducible generation
- `--verbose`, `-v`: Enable verbose output

### Examples

```bash
# Basic generation
python generate_stripe_masks.py -o ./masks -s 128 -n 100

# Reproducible generation with seed
python generate_stripe_masks.py -o ./masks -s 256 -n 50 --seed 42

# Custom prefix and verbose output
python generate_stripe_masks.py -o ./training_data -s 512 -n 10 -p "custom_stripe" -v
```

## Output Format

### File Structure
- Files are saved as `{prefix}_{index:04d}.tif`
- Example: `stripe_mask_0001.tif`, `stripe_mask_0002.tif`, etc.

### Mask Values
- **0**: Valid/unmasked data areas (black)
- **255**: Masked/missing data areas (white)

### File Properties
- Format: GeoTIFF
- Data type: 8-bit unsigned integer (uint8)
- Compression: LZW (lossless)
- Single band (grayscale)

## Pattern Specifications

### Stripe Parameters
- **Masked stripe width**: Randomly varies from 20-90 pixels per tile
- **Unmasked stripe width**: Randomly varies from 40-100 pixels per tile
- **Rotation angle**: Randomly varies from 0-360 degrees per tile
- **Pattern type**: Parallel stripes with alternating masked/unmasked regions

### Coverage Statistics
Typical mask coverage ranges from 20-70% depending on random parameters:
- Lower coverage: Wide unmasked stripes + narrow masked stripes
- Higher coverage: Narrow unmasked stripes + wide masked stripes

## Integration with Training Workflows

### Use with DEM Inpainting Models
These masks can be used directly with the existing DEM inpainting codebase:

1. **Generate masks**: Use this script to create training masks
2. **Pair with data**: Match mask files with corresponding DEM tiles
3. **Dataset loading**: The existing dataset loaders can use these masks
4. **Training**: Use for supervised learning of inpainting tasks

### File Naming for Integration
For compatibility with existing workflows, consider these naming patterns:
```bash
# For tile-based training
python generate_stripe_masks.py -o ./masks -p "mask" -n 1000

# For specific dataset integration
python generate_stripe_masks.py -o ./dataset/masks -p "stripe_mask" -n 500
```

## Quality Assurance

The script provides automatic statistics after generation:
- **Mask coverage ratio**: Percentage of masked pixels
- **Stripe width statistics**: Average, min, and max stripe widths
- **File generation confirmation**: Lists all created files

## Limitations

1. **Fixed stripe orientation**: All stripes within a single mask are parallel
2. **Binary masks only**: No support for partial masking (0-255 values)
3. **Square tiles only**: Rectangular tiles are not supported
4. **No georeferencing**: Masks are generated without spatial reference

## Troubleshooting

### Common Issues

**Unicode encoding errors on Windows:**
- The script uses ASCII characters for compatibility with Windows terminals

**Missing dependencies:**
```bash
pip install numpy pillow rasterio
```

**Permission errors:**
- Ensure the output directory is writable
- Check that existing files can be overwritten

### File Size Considerations
- 128x128 masks: ~1KB per file
- 256x256 masks: ~2-4KB per file  
- 512x512 masks: ~8-16KB per file (varies with stripe pattern)

## Example Output

When generating 3 masks with the command:
```bash
python generate_stripe_masks.py --output_dir ./test_masks --tile_size 128 --num_tiles 3 --seed 42
```

Expected output:
```
Random seed set to: 42
Generating 3 stripe masks...
Tile size: 128x128 pixels
Output directory: ./test_masks
Mask stripe width range: 20-90 pixels
Unmasked stripe width range: 40-100 pixels
Rotation: 0-360 degrees (random)

Generated 1/3 masks... (stripe_mask_0001.tif)
Generated 3/3 masks... (stripe_mask_0003.tif)

------------------------------------------------------------
GENERATION COMPLETE
------------------------------------------------------------
Total files generated: 3
Output directory: ./test_masks

Mask coverage statistics:
  Average mask ratio: 0.380 (38.0%)
  Min mask ratio: 0.304 (30.4%)
  Max mask ratio: 0.476 (47.6%)

Stripe width statistics:
  Masked stripes - Avg: 39.3px, Range: 33-51px
  Unmasked stripes - Avg: 59.3px, Range: 41-83px

Success! Generated 3 stripe masks in ./test_masks