#!/usr/bin/env python3
"""
Generate stripe patterned mask files for training neural networks.

This script creates square TIF mask files with parallel stripe patterns where:
- Masked areas (value 255) represent missing/nodata regions
- Unmasked areas (value 0) represent valid data regions

The stripes are randomly rotated and have varying widths to create diverse
training masks for DEM inpainting and similar tasks.

Usage:
    python generate_stripe_masks.py --output_dir ./masks --tile_size 128 --num_tiles 100
    python generate_stripe_masks.py --output_dir ./training_masks --tile_size 256 --num_tiles 50 --prefix "stripe_mask"
"""

import argparse
import os
import sys
import random
import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.transform import from_bounds
import math


def create_stripe_pattern(size, mask_stripe_width, unmasked_stripe_width, rotation_angle):
    """
    Create a stripe pattern with specified parameters.
    
    Args:
        size (int): Square image size in pixels
        mask_stripe_width (int): Width of masked stripes (10-110px)
        unmasked_stripe_width (int): Width of unmasked stripes (10-110px)
        rotation_angle (float): Rotation angle in degrees (0-360)
    
    Returns:
        numpy.ndarray: 2D array with stripe pattern (0=valid, 255=masked)
    """
    # Create initial stripe pattern
    pattern_width = mask_stripe_width + unmasked_stripe_width
    
    # Create a larger canvas to handle rotation without clipping
    diagonal = int(size * math.sqrt(2)) + pattern_width
    large_size = diagonal + pattern_width
    
    # Generate horizontal stripes on larger canvas
    pattern = np.zeros((large_size, large_size), dtype=np.uint8)
    
    # Fill with alternating stripes
    y = 0
    masked = True  # Start with masked stripe
    
    while y < large_size:
        if masked:
            # Add masked stripe (value 255)
            pattern[y:y+mask_stripe_width, :] = 255
            y += mask_stripe_width
        else:
            # Add unmasked stripe (value 0) - already zeros
            y += unmasked_stripe_width
        masked = not masked
    
    # Convert to PIL image for rotation
    img = Image.fromarray(pattern, mode='L')
    
    # Rotate the pattern
    rotated_img = img.rotate(rotation_angle, expand=False, fillcolor=0)
    
    # Convert back to numpy array
    rotated_pattern = np.array(rotated_img)
    
    # Crop to desired size from center
    center_y, center_x = large_size // 2, large_size // 2
    half_size = size // 2
    
    y_start = center_y - half_size
    y_end = center_y + half_size
    x_start = center_x - half_size
    x_end = center_x + half_size
    
    # Ensure we don't go out of bounds
    y_start = max(0, y_start)
    y_end = min(large_size, y_end)
    x_start = max(0, x_start)
    x_end = min(large_size, x_end)
    
    cropped_pattern = rotated_pattern[y_start:y_end, x_start:x_end]
    
    # Pad if necessary to ensure exact size
    if cropped_pattern.shape[0] != size or cropped_pattern.shape[1] != size:
        result = np.zeros((size, size), dtype=np.uint8)
        h, w = cropped_pattern.shape
        start_y = (size - h) // 2
        start_x = (size - w) // 2
        result[start_y:start_y+h, start_x:start_x+w] = cropped_pattern
        return result
    
    return cropped_pattern


def generate_single_mask(size, output_path):
    """
    Generate a single stripe mask with random parameters.
    
    Args:
        size (int): Square image size in pixels
        output_path (str): Path to save the generated mask
    """
    # Random parameters within specified ranges
    mask_stripe_width = random.randint(10, 110)
    unmasked_stripe_width = random.randint(10, 110)
    rotation_angle = random.uniform(0, 360)
    
    # Generate the stripe pattern
    pattern = create_stripe_pattern(size, mask_stripe_width, unmasked_stripe_width, rotation_angle)
    
    # Save as TIF file using rasterio for consistency
    profile = {
        'driver': 'GTiff',
        'height': size,
        'width': size,
        'count': 1,
        'dtype': 'uint8',
        'compress': 'lzw',
        'nodata': None
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(pattern, 1)
    
    return {
        'mask_stripe_width': mask_stripe_width,
        'unmasked_stripe_width': unmasked_stripe_width,
        'rotation_angle': rotation_angle,
        'masked_pixels': int(np.sum(pattern == 255)),
        'unmasked_pixels': int(np.sum(pattern == 0)),
        'mask_ratio': np.sum(pattern == 255) / (size * size)
    }


def generate_masks(output_dir, tile_size, num_tiles, prefix="stripe_mask"):
    """
    Generate multiple stripe mask files.
    
    Args:
        output_dir (str): Directory to save generated masks
        tile_size (int): Square tile size in pixels
        num_tiles (int): Number of mask tiles to generate
        prefix (str): Filename prefix for generated masks
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_tiles} stripe masks...")
    print(f"Tile size: {tile_size}x{tile_size} pixels")
    print(f"Output directory: {output_dir}")
    print(f"Mask stripe width range: 10-110 pixels")
    print(f"Unmasked stripe width range: 10-110 pixels")
    print(f"Rotation: 0-360 degrees (random)")
    print()
    
    statistics = []
    
    for i in range(num_tiles):
        # Generate filename
        filename = f"{prefix}_{i+1:04d}.tif"
        output_path = os.path.join(output_dir, filename)
        
        # Generate mask
        stats = generate_single_mask(tile_size, output_path)
        stats['filename'] = filename
        statistics.append(stats)
        
        # Progress update
        if (i + 1) % 10 == 0 or i == 0 or i == num_tiles - 1:
            print(f"Generated {i+1}/{num_tiles} masks... ({filename})")
    
    # Print summary statistics
    print("\n" + "-" * 60)
    print("GENERATION COMPLETE")
    print("-" * 60)
    
    mask_ratios = [s['mask_ratio'] for s in statistics]
    mask_stripe_widths = [s['mask_stripe_width'] for s in statistics]
    unmasked_stripe_widths = [s['unmasked_stripe_width'] for s in statistics]
    
    print(f"Total files generated: {len(statistics)}")
    print(f"Output directory: {output_dir}")
    print(f"\nMask coverage statistics:")
    print(f"  Average mask ratio: {np.mean(mask_ratios):.3f} ({np.mean(mask_ratios)*100:.1f}%)")
    print(f"  Min mask ratio: {np.min(mask_ratios):.3f} ({np.min(mask_ratios)*100:.1f}%)")
    print(f"  Max mask ratio: {np.max(mask_ratios):.3f} ({np.max(mask_ratios)*100:.1f}%)")
    print(f"\nStripe width statistics:")
    print(f"  Masked stripes - Avg: {np.mean(mask_stripe_widths):.1f}px, Range: {np.min(mask_stripe_widths)}-{np.max(mask_stripe_widths)}px")
    print(f"  Unmasked stripes - Avg: {np.mean(unmasked_stripe_widths):.1f}px, Range: {np.min(unmasked_stripe_widths)}-{np.max(unmasked_stripe_widths)}px")


def main():
    """Main function to handle command line arguments and execute mask generation."""
    
    parser = argparse.ArgumentParser(
        description='Generate stripe patterned mask files for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_stripe_masks.py --output_dir ./masks --tile_size 128 --num_tiles 100
  python generate_stripe_masks.py --output_dir ./training_masks --tile_size 256 --num_tiles 50
  python generate_stripe_masks.py --output_dir ./masks_512 --tile_size 512 --num_tiles 25 --prefix "large_stripe"

Stripe Parameters:
  - Masked stripe width: randomly varies from 10-110 pixels
  - Unmasked stripe width: randomly varies from 10-110 pixels
  - Rotation angle: randomly varies from 0-360 degrees
  - Pattern: parallel stripes with random rotation per tile

Output:
  - Square TIF files with binary masks
  - Value 255 = masked/missing data areas
  - Value 0 = valid/unmasked data areas
  - Lossless LZW compression
        """
    )
    
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                       help='Output directory for generated mask files')
    parser.add_argument('--tile_size', '-s', type=int, choices=[128, 256, 512], default=128,
                       help='Square tile size in pixels (default: 128)')
    parser.add_argument('--num_tiles', '-n', type=int, required=True,
                       help='Number of mask tiles to generate')
    parser.add_argument('--prefix', '-p', type=str, default='stripe_mask',
                       help='Filename prefix for generated masks (default: stripe_mask)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Validate arguments
    if args.num_tiles <= 0:
        print("Error: Number of tiles must be positive")
        sys.exit(1)
    
    if not args.output_dir:
        print("Error: Output directory must be specified")
        sys.exit(1)
    
    try:
        # Generate the masks
        generate_masks(args.output_dir, args.tile_size, args.num_tiles, args.prefix)
        
        print(f"\nSuccess! Generated {args.num_tiles} stripe masks in {args.output_dir}")
        
    except Exception as e:
        print(f"\nError generating masks: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()