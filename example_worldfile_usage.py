#!/usr/bin/env python3
"""
Example usage of the tile world file generator.

This script demonstrates how to generate world files for tiles and verify they work correctly.
"""

import os
import sys
from pathlib import Path
from generate_tile_worldfiles import generate_worldfiles_for_tiles, parse_tile_filename
import rasterio
import rasterio.crs

def example_usage():
    """Example of how to use the world file generator."""
    
    print("=== Tile World File Generator Example ===\n")
    
    # Example parameters (adjust these to your actual data)
    input_image = "./test/21-small.tif"  # Original georeferenced DEM
    tiles_directory = "./experiments/test_DEM_inpainting_251030_165439/results/test/0"  # Directory with tiles
    tile_size = 128
    overlap = 0
    prefix = "tile"
    
    print(f"Input image: {input_image}")
    print(f"Tiles directory: {tiles_directory}")
    print(f"Tile size: {tile_size}x{tile_size}")
    print(f"Overlap: {overlap} pixels")
    print(f"Prefix: {prefix}")
    print()
    
    # Check if input image exists
    if not os.path.exists(input_image):
        print(f"Warning: Input image {input_image} not found.")
        print("Please adjust the path to point to your original georeferenced image.")
        return
    
    # Check if tiles directory exists
    if not os.path.exists(tiles_directory):
        print(f"Warning: Tiles directory {tiles_directory} not found.")
        print("Please adjust the path to point to your tiles directory.")
        return
    
    try:
        # Generate world files
        print("Generating world files...")
        generate_worldfiles_for_tiles(
            input_image_path=input_image,
            tiles_directory=tiles_directory,
            tile_size=tile_size,
            overlap=overlap,
            prefix=prefix,
            force_overwrite=True  # Overwrite existing files for this example
        )
        
        print("\n=== Verification ===")
        verify_worldfiles(tiles_directory, prefix)
        
    except Exception as e:
        print(f"Error: {e}")


def verify_worldfiles(tiles_directory: str, prefix: str = "tile"):
    """Verify that world files were created and can be read correctly."""
    
    tiles_dir = Path(tiles_directory)
    tile_files = list(tiles_dir.glob(f"{prefix}_*.tif"))
    world_files = list(tiles_dir.glob(f"{prefix}_*.twf"))
    
    print(f"Found {len(tile_files)} tile files")
    print(f"Found {len(world_files)} world files")
    
    if len(world_files) == 0:
        print("No world files found!")
        return
    
    # Check a few world files
    for i, world_file in enumerate(sorted(world_files)[:3]):
        print(f"\nChecking {world_file.name}:")
        
        try:
            # Read world file content
            with open(world_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) != 6:
                print(f"  Error: Expected 6 lines, found {len(lines)}")
                continue
            
            # Parse world file parameters
            pixel_size_x = float(lines[0].strip())
            rotation_y = float(lines[1].strip())
            rotation_x = float(lines[2].strip())
            pixel_size_y = float(lines[3].strip())
            x_center = float(lines[4].strip())
            y_center = float(lines[5].strip())
            
            print(f"  Pixel size X: {pixel_size_x:.6f}")
            print(f"  Rotation Y: {rotation_y:.6f}")
            print(f"  Rotation X: {rotation_x:.6f}")
            print(f"  Pixel size Y: {pixel_size_y:.6f}")
            print(f"  X center: {x_center:.6f}")
            print(f"  Y center: {y_center:.6f}")
            
            # Try to open the corresponding tile with rasterio to see if it works
            tile_file = world_file.with_suffix('.tif')
            if tile_file.exists():
                try:
                    with rasterio.open(tile_file) as src:
                        print(f"  Tile dimensions: {src.width}x{src.height}")
                        print(f"  Tile CRS: {src.crs}")
                        if src.transform:
                            print(f"  Tile transform: {src.transform}")
                except Exception as e:
                    print(f"  Could not read tile: {e}")
            
        except Exception as e:
            print(f"  Error reading world file: {e}")


def test_filename_parsing():
    """Test the filename parsing function."""
    
    print("\n=== Testing Filename Parsing ===")
    
    test_cases = [
        ("tile_0001_0002.tif", "tile", (1, 2)),
        ("mask_0003_0004.tif", "mask", (3, 4)),
        ("GT_tile_0005_0006.tif", "tile", None),  # Wrong prefix
        ("tile_001_002.tif", "tile", None),  # Wrong format
        ("notile_0007_0008.tif", "tile", None),  # Wrong prefix
    ]
    
    for filename, prefix, expected in test_cases:
        result = parse_tile_filename(filename, prefix)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {filename} (prefix='{prefix}') -> {result} (expected: {expected})")


if __name__ == "__main__":
    # Run tests
    test_filename_parsing()
    
    # Run example
    example_usage()
    
    print("\n=== Usage Instructions ===")
    print("To use the world file generator:")
    print("1. Run inference to generate tiles")
    print("2. Generate world files:")
    print("   python generate_tile_worldfiles.py --input /path/to/original.tif --tiles_dir /path/to/tiles")
    print("3. Load tiles in GIS software - they should be properly georeferenced!")
    print("\nCommand line options:")
    print("  --tile_size: Size of tiles (default: 128)")
    print("  --overlap: Overlap between tiles (default: 0)")
    print("  --prefix: Tile filename prefix (default: 'tile')")
    print("  --force: Overwrite existing world files")