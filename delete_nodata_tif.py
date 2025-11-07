#!/usr/bin/env python3
"""
Simple script to delete TIFF files containing nodata pixels.
Usage: python delete_nodata_tif.py <path_to_tif_file>
"""

import sys
import os
import rasterio
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("Usage: python delete_nodata_tif.py <path_to_tif_file>")
        sys.exit(1)
    
    tif_file = sys.argv[1]
    
    if not os.path.exists(tif_file):
        print(f"Error: File {tif_file} does not exist")
        sys.exit(1)
    
    try:
        with rasterio.open(tif_file) as src:
            data = src.read(1)
            
            # Check for nodata pixels with value -99999
            if np.any(data == -99999):
                print(f"Deleting {tif_file} (contains nodata pixels)")
                os.remove(tif_file)
            else:
                print(f"Keeping {tif_file} (no nodata pixels)")
    
    except Exception as e:
        print(f"Error processing {tif_file}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()