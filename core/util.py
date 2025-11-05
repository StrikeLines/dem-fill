import random
import numpy as np
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
import torch.nn.functional as F
import json
import rasterio
import rasterio.transform
import rasterio.crs
from typing import Dict, Tuple
import os
from rasterio.windows import Window
from rasterio.transform import from_bounds
  

class GeoTiffTiler:
    def __init__(self, tile_size: int = 128, overlap: int = 0):
        """
        Initialize the GeoTiff tiler
        
        Args:
            tile_size: Size of square tiles (default: 128)
            overlap: Overlap between tiles in pixels (default: 0)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.tiles_dir = None
        self.metadata_path = None
        
    def makeflist(self, metadata: Dict, tiles_dir: str) -> str:
        """
        Create a file list from metadata for dataset loading
        
        Args:
            metadata: Tiling metadata dictionary
            tiles_dir: Directory containing the tiles
            
        Returns:
            Path to the created file list
        """
        flist_path = os.path.join(tiles_dir, "filelist.txt")
        
        with open(flist_path, 'w') as f:
            for tile in metadata['tiles']:
                tile_path = os.path.join(tiles_dir, tile['filename'])
                f.write(f"{tile_path}\n")
        
        return flist_path
    
    def save_metadata(self, metadata: Dict, filepath: str) -> None:
        """Save metadata to a file for later use"""
        
        # Convert non-serializable objects
        serializable_metadata = metadata.copy()
        serializable_metadata['transform'] = list(metadata['transform'])
        
        # Better CRS serialization - use to_wkt() or to_dict() for more reliable parsing
        if metadata['crs'] is not None:
            try:
                serializable_metadata['crs'] = metadata['crs'].to_wkt()
                serializable_metadata['crs_type'] = 'wkt'
            except:
                try:
                    serializable_metadata['crs'] = metadata['crs'].to_dict()
                    serializable_metadata['crs_type'] = 'dict'
                except:
                    serializable_metadata['crs'] = str(metadata['crs'])
                    serializable_metadata['crs_type'] = 'string'
        else:
            serializable_metadata['crs'] = None
            serializable_metadata['crs_type'] = 'none'
        
        # Convert tile transforms
        for tile in serializable_metadata['tiles']:
            tile['transform'] = list(tile['transform'])
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        
        self.metadata_path = filepath

    def load_metadata(self, filepath: str = None) -> Dict:
        """Load metadata from a file"""
        
        if filepath is None:
            filepath = self.metadata_path
            
        if filepath is None:
            raise ValueError("No metadata filepath provided and none stored in class")
        
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        # Convert back to rasterio objects
        metadata['transform'] = rasterio.transform.Affine(*metadata['transform'])
        
        # Better CRS deserialization based on how it was saved
        crs_type = metadata.get('crs_type', 'string')
        if metadata['crs'] is not None:
            if crs_type == 'wkt':
                metadata['crs'] = rasterio.crs.CRS.from_wkt(metadata['crs'])
            elif crs_type == 'dict':
                metadata['crs'] = rasterio.crs.CRS.from_dict(metadata['crs'])
            else:  # string or fallback
                try:
                    metadata['crs'] = rasterio.crs.CRS.from_string(metadata['crs'])
                except:
                    print(f"Warning: Could not parse CRS '{metadata['crs']}', setting to None")
                    metadata['crs'] = None
        else:
            metadata['crs'] = None
        
        # Convert tile transforms
        for tile in metadata['tiles']:
            tile['transform'] = rasterio.transform.Affine(*tile['transform'])
        
        return metadata

    def tile_image(self, input_path: str, output_dir: str, prefix: str = "tile") -> Dict:
        """
        Tile a georeferenced image into smaller patches with overlap support
        
        Args:
            input_path: Path to input GeoTIFF
            output_dir: Directory to save tiles
            prefix: Prefix for tile filenames
            
        Returns:
            Dictionary with tiling metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        
        with rasterio.open(input_path) as src:
            # Get image dimensions and geospatial info
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
            dtype = src.dtypes[0]
            count = src.count
            
            # Calculate effective step size (tile_size - overlap)
            step_size = self.tile_size - self.overlap
            
            # Calculate number of tiles needed with overlap
            tiles_x = math.ceil((width - self.overlap) / step_size)
            tiles_y = math.ceil((height - self.overlap) / step_size)
            
            # Calculate padded dimensions if needed
            padded_width = (tiles_x - 1) * step_size + self.tile_size
            padded_height = (tiles_y - 1) * step_size + self.tile_size
            
            print(f"Original size: {width}x{height}")
            print(f"Tile size: {self.tile_size}x{self.tile_size}")
            print(f"Overlap: {self.overlap} pixels")
            print(f"Step size: {step_size} pixels")
            print(f"Padded size: {padded_width}x{padded_height}")
            print(f"Number of tiles: {tiles_x}x{tiles_y} = {tiles_x * tiles_y}")
            
            metadata = {
                'original_width': width,
                'original_height': height,
                'padded_width': padded_width,
                'padded_height': padded_height,
                'tiles_x': tiles_x,
                'tiles_y': tiles_y,
                'tile_size': self.tile_size,
                'overlap': self.overlap,
                'step_size': step_size,
                'transform': transform,
                'crs': crs,
                'nodata': nodata,
                'dtype': dtype,
                'count': count,
                'tiles': []
            }
            
            # Generate tiles with overlap
            for row in range(tiles_y):
                for col in range(tiles_x):
                    # Calculate tile bounds with overlap
                    x_start = col * step_size
                    y_start = row * step_size
                    x_end = min(x_start + self.tile_size, width)
                    y_end = min(y_start + self.tile_size, height)
                    
                    # Read the actual data within image bounds
                    actual_width = x_end - x_start
                    actual_height = y_end - y_start
                    
                    window = Window(x_start, y_start, actual_width, actual_height)
                    tile_data = src.read(window=window)
                    
                    # Create padded tile if necessary
                    if actual_width < self.tile_size or actual_height < self.tile_size:
                        # Create padded tile using edge extension instead of nodata
                        padded_tile = np.zeros((count, self.tile_size, self.tile_size), dtype=tile_data.dtype)
                        
                        # Copy actual data
                        padded_tile[:, :actual_height, :actual_width] = tile_data
                        
                        # Pad using edge extension (repeat edge values)
                        # Right padding (extend rightmost column)
                        if actual_width < self.tile_size:
                            padded_tile[:, :actual_height, actual_width:] = \
                                tile_data[:, :actual_height, -1:].repeat(self.tile_size - actual_width, axis=2)
                        
                        # Bottom padding (extend bottom row)
                        if actual_height < self.tile_size:
                            padded_tile[:, actual_height:, :] = \
                                padded_tile[:, actual_height-1:actual_height, :].repeat(self.tile_size - actual_height, axis=1)
                        
                        tile_data = padded_tile
                    
                    # Calculate tile's geotransform
                    left, top = transform * (x_start, y_start)
                    right, bottom = transform * (x_start + self.tile_size, y_start + self.tile_size)
                    
                    tile_transform = rasterio.transform.from_bounds(
                        left, bottom, right, top,  # west, south, east, north
                        self.tile_size, self.tile_size
                    )
                    
                    # Save tile
                    tile_filename = f"{prefix}_{row:04d}_{col:04d}.tif"
                    tile_path = os.path.join(output_dir, tile_filename)
                    
                    with rasterio.open(
                        tile_path, 'w',
                        driver='GTiff',
                        height=self.tile_size,
                        width=self.tile_size,
                        count=count,
                        dtype=dtype,
                        crs=crs,
                        transform=tile_transform,
                        nodata=nodata
                    ) as dst:
                        dst.write(tile_data)
                    
                    tile_info = {
                        'filename': tile_filename,
                        'row': row,
                        'col': col,
                        'x_start': x_start,
                        'y_start': y_start,
                        'actual_width': actual_width,
                        'actual_height': actual_height,
                        'transform': tile_transform
                    }
                    metadata['tiles'].append(tile_info)
                    
                    if (row * tiles_x + col + 1) % 50 == 0:
                        print(f"Processed {row * tiles_x + col + 1}/{tiles_x * tiles_y} tiles")
        
        self.tiles_dir = output_dir
        return metadata
    
    def tile_image_with_mask(
        self,
        input_path: str,
        tiles_dir: str,
        masks_dir: str,
        prefix_img: str = "tile",
        prefix_mask: str = "mask",
        nodata_value: float = None
    ) -> Tuple[Dict, Dict]:
        """
        Tile a georeferenced DEM into image and mask tiles.
        Automatically detects or uses user-provided NoData value.

        Args:
            input_path: Path to input GeoTIFF (DEM)
            tiles_dir: Output folder for image tiles
            masks_dir: Output folder for mask tiles
            prefix_img: Prefix for tile filenames
            prefix_mask: Prefix for mask filenames
            nodata_value: Optional NoData value (auto-detected if None)

        Returns:
            (tile_metadata, mask_metadata): Two dictionaries (image + mask)
        """
        import math
        os.makedirs(tiles_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        with rasterio.open(input_path) as src:
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
            nodata = src.nodata if src.nodata is not None else nodata_value
            dtype = src.dtypes[0]
            count = src.count

            # Auto-detect nodata if not specified
            if nodata is None:
                arr_sample = src.read(1, masked=True)
                candidates = [-9999, -32768, 9999, 1e30, 3.4028235e38]
                counts = {v: np.sum(arr_sample.data == v) for v in candidates}
                nodata = max(counts, key=counts.get)
                if counts[nodata] == 0:
                    nodata = None
                print(f"Auto-detected NoData value: {nodata}")

            # print(f"detected NoData value: {nodata}")

            step_size = self.tile_size - self.overlap
            tiles_x = math.ceil((width - self.overlap) / step_size)
            tiles_y = math.ceil((height - self.overlap) / step_size)
            padded_width = (tiles_x - 1) * step_size + self.tile_size
            padded_height = (tiles_y - 1) * step_size + self.tile_size

            print(f"Original size: {width}x{height}")
            print(f"Tile size: {self.tile_size}x{self.tile_size}")
            print(f"Overlap: {self.overlap} pixels")
            print(f"Step size: {step_size} pixels")
            print(f"Padded size: {padded_width}x{padded_height}")
            print(f"Number of tiles: {tiles_x}x{tiles_y} = {tiles_x * tiles_y}")

            # Metadata
            def base_meta():
                return {
                    'original_width': width,
                    'original_height': height,
                    'padded_width': padded_width,
                    'padded_height': padded_height,
                    'tiles_x': tiles_x,
                    'tiles_y': tiles_y,
                    'tile_size': self.tile_size,
                    'overlap': self.overlap,
                    'step_size': step_size,
                    'transform': transform,
                    'crs': crs,
                    'nodata': nodata,
                    'dtype': dtype,
                    'count': count,
                    'tiles': []
                }

            tile_metadata = base_meta()
            mask_metadata = base_meta()

            # Generate tiles + masks
            for row in range(tiles_y):
                for col in range(tiles_x):
                    x_start = col * step_size
                    y_start = row * step_size
                    x_end = min(x_start + self.tile_size, width)
                    y_end = min(y_start + self.tile_size, height)

                    actual_width = x_end - x_start
                    actual_height = y_end - y_start

                    window = Window(x_start, y_start, actual_width, actual_height)
                    tile_data = src.read(window=window)

                    # Pad edges if smaller than tile size
                    if actual_width < self.tile_size or actual_height < self.tile_size:
                        padded_tile = np.zeros((count, self.tile_size, self.tile_size), dtype=tile_data.dtype)
                        padded_tile[:, :actual_height, :actual_width] = tile_data

                        # Edge extension padding
                        if actual_width < self.tile_size:
                            padded_tile[:, :actual_height, actual_width:] = \
                                tile_data[:, :actual_height, -1:].repeat(self.tile_size - actual_width, axis=2)
                        if actual_height < self.tile_size:
                            padded_tile[:, actual_height:, :] = \
                                padded_tile[:, actual_height-1:actual_height, :].repeat(self.tile_size - actual_height, axis=1)

                        tile_data = padded_tile

                    # Build mask (255 for NoData)
                    if nodata is not None:
                        # print(f"noData is not None: {nodata}")

                        mask = np.where(tile_data == nodata, 255, 0).astype(np.uint8)[0]
                        tile_data = np.where(tile_data == nodata, 0, tile_data)
                    else:
                        print("noData is None making all zeros mask")
                        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)

                    # Compute geotransform
                    left, top = transform * (x_start, y_start)
                    right, bottom = transform * (x_start + self.tile_size, y_start + self.tile_size)
                    tile_transform = rasterio.transform.from_bounds(left, bottom, right, top, self.tile_size, self.tile_size)

                    # Write DEM tile
                    tile_name = f"{prefix_img}_{row:04d}_{col:04d}.tif"
                    mask_name = f"{prefix_mask}_{row:04d}_{col:04d}.tif"
                    tile_path = os.path.join(tiles_dir, tile_name)
                    mask_path = os.path.join(masks_dir, mask_name)

                    with rasterio.open(
                        tile_path, 'w',
                        driver='GTiff',
                        height=self.tile_size,
                        width=self.tile_size,
                        count=count,
                        dtype=dtype,
                        crs=crs,
                        transform=tile_transform,
                        nodata=nodata
                    ) as dst:
                        dst.write(tile_data)

                    with rasterio.open(
                        mask_path, 'w',
                        driver='GTiff',
                        height=self.tile_size,
                        width=self.tile_size,
                        count=1,
                        dtype='uint8',
                        crs=crs,
                        transform=tile_transform
                    ) as dst:
                        dst.write(mask, 1)  # ✅ ensures correct shape (bands, H, W)

                    tile_metadata['tiles'].append({
                        'filename': tile_name,
                        'row': row,
                        'col': col,
                        'x_start': x_start,
                        'y_start': y_start,
                        'actual_width': actual_width,
                        'actual_height': actual_height,
                        'transform': tile_transform
                    })
                    mask_metadata['tiles'].append({
                        'filename': mask_name,
                        'row': row,
                        'col': col,
                        'x_start': x_start,
                        'y_start': y_start,
                        'actual_width': actual_width,
                        'actual_height': actual_height,
                        'transform': tile_transform
                    })

                    if (row * tiles_x + col + 1) % 50 == 0:
                        print(f"Processed {row * tiles_x + col + 1}/{tiles_x * tiles_y} tiles")

        print(f"✅ Done: {len(tile_metadata['tiles'])} tiles written to {tiles_dir} and {masks_dir}")
        return tile_metadata, mask_metadata

    def merge_tiles(self, tiles_dir: str, metadata: Dict, output_path: str, 
                   crop_to_original: bool = True) -> None:
        """
        Merge overlapping tiles back into a single georeferenced image with averaging
        
        Args:
            tiles_dir: Directory containing tiles
            metadata: Metadata from tiling process
            output_path: Path for output merged image
            crop_to_original: Whether to crop back to original dimensions
        """
        tiles_x = metadata['tiles_x']
        tiles_y = metadata['tiles_y']
        count = metadata['count']
        dtype = metadata['dtype']
        nodata = metadata['nodata']
        overlap = metadata.get('overlap', 0)
        step_size = metadata.get('step_size', metadata['tile_size'])
        
        # Determine output dimensions
        if crop_to_original:
            out_width = metadata['original_width']
            out_height = metadata['original_height']
        else:
            out_width = metadata['padded_width']
            out_height = metadata['padded_height']
        
        # Create output arrays for data and count (for averaging)
        merged_data = np.zeros((count, out_height, out_width), dtype=np.float64)
        count_array = np.zeros((out_height, out_width), dtype=np.int32)
        
        print(f"Merging {len(metadata['tiles'])} tiles with overlap averaging...")
        if overlap > 0:
            print(f"Overlap: {overlap} pixels, averaging overlapping regions")
        
        # Process each tile
        for i, tile_info in enumerate(metadata['tiles']):
            tile_path = os.path.join(tiles_dir, tile_info['filename'])
            
            if not os.path.exists(tile_path):
                print(f"Warning: Tile {tile_info['filename']} not found, skipping")
                continue
            
            with rasterio.open(tile_path) as src:
                tile_data = src.read().astype(np.float64)
            
            # Calculate where to place this tile in the merged image
            row, col = tile_info['row'], tile_info['col']
            
            # Calculate position based on step size for overlapping tiles
            x_start = col * step_size
            y_start = row * step_size
            
            # Calculate actual area to copy (handle edge tiles and cropping)
            if crop_to_original:
                x_end = min(x_start + metadata['tile_size'], metadata['original_width'])
                y_end = min(y_start + metadata['tile_size'], metadata['original_height'])
            else:
                x_end = min(x_start + metadata['tile_size'], out_width)
                y_end = min(y_start + metadata['tile_size'], out_height)
            
            copy_width = x_end - x_start
            copy_height = y_end - y_start
            
            # Create a mask for valid data (non-nodata values)
            if nodata is not None:
                valid_mask = tile_data != nodata
            else:
                valid_mask = np.ones_like(tile_data, dtype=bool)
            
            # Add tile data to merged array and increment count for valid pixels
            for c in range(count):
                tile_slice = tile_data[c, :copy_height, :copy_width]
                valid_slice = valid_mask[c, :copy_height, :copy_width]
                
                # Only add valid pixels
                merged_data[c, y_start:y_end, x_start:x_end][valid_slice] += tile_slice[valid_slice]
                count_array[y_start:y_end, x_start:x_end][valid_slice] += 1
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(metadata['tiles'])} tiles")
        
        # Average the overlapping pixels
        print("Computing averages for overlapping regions...")
        
        # Avoid division by zero
        count_array[count_array == 0] = 1
        
        # Average the accumulated values
        for c in range(count):
            merged_data[c] = np.divide(merged_data[c], count_array, 
                                     out=np.zeros_like(merged_data[c]), 
                                     where=count_array!=0)
        
        # Set nodata values where no tiles contributed
        if nodata is not None:
            no_data_mask = (count_array == 1) & (merged_data[0] == 0)  # Areas with no valid data
            merged_data[:, no_data_mask] = nodata
        
        # Convert back to original dtype
        if dtype != np.float64:
            merged_data = merged_data.astype(dtype)
        
        # Save merged image
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=out_height,
            width=out_width,
            count=count,
            dtype=dtype,
            crs=metadata['crs'],
            transform=metadata['transform'],
            nodata=nodata,
            compress='lzw'
        ) as dst:
            dst.write(merged_data)
        
        print(f"Merged image saved to: {output_path}")
        print(f"Final dimensions: {out_width}x{out_height}")
        if overlap > 0:
            overlap_pixels = np.sum(count_array > 1)
            total_pixels = count_array.size
            print(f"Overlapping pixels averaged: {overlap_pixels:,}/{total_pixels:,} ({100*overlap_pixels/total_pixels:.1f}%)")

## Changed Here! replaced tensor2img
# def tensor2img(tensor, min_max=(-1, 1), out_type=np.uint8, scale_factor=1):
#     '''
#     Converts a torch Tensor into an image Numpy array
#     Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
#     Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
#     '''
#     # tensor = tensor.clamp_(*min_max)  # clamp
#     n_dim = tensor.dim()
#     if n_dim == 4:
#         n_img = len(tensor)
#         img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
#         img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
#     elif n_dim == 3:
#         img_np = tensor.numpy()
#         img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
#     elif n_dim == 2:
#         img_np = tensor.numpy()
#     else:
#         raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    
#     if out_type == np.uint8:
#         img_np = ((img_np+1) * 127.5).round()
#         # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
#     elif out_type == np.uint16:
#         img_np = ((img_np+1) / 2 * scale_factor * 256).round()

#     return img_np.astype(out_type).squeeze()

## Changed here! With this
def tensor2img(tensor, min_max=(-1, 1), out_type=np.uint8, scale_factor=1, min_val=None, max_val=None):
    '''
    Converts a torch Tensor into an image Numpy array.
    Simplified version for consistent tile processing to prevent edge artifacts.
    
    Input:
        - tensor: torch.Tensor (4D, 3D, or 2D)
        - min_max: Tuple (min, max), assumed normalization range (default: [-1, 1])
        - out_type: np.uint8, np.uint16, or np.float32
        - scale_factor: scale multiplier for uint16 conversion (default: 1)
        - min_val, max_val: optional original DEM min/max for true denormalization

    Output:
        - Numpy array image (HWC or HW) in specified format
    '''
    import math
    from torchvision.utils import make_grid
    tensor = tensor.clone().detach()
    tensor = torch.clamp(tensor, min=min_max[0], max=min_max[1])

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(f'Only support 4D, 3D and 2D tensor. But received dimension: {n_dim}')

    # Convert to [0, 1] from given min_max range
    img_np = (img_np - min_max[0]) / (min_max[1] - min_max[0])
    img_np = np.clip(img_np, 0, 1)

    # Simplified tensor handling to ensure consistency across tiles
    if min_val is not None and max_val is not None:
        if isinstance(min_val, torch.Tensor):
            min_val = min_val.item()
        
        if isinstance(max_val, torch.Tensor):
            max_val = max_val.item()

        img_np = img_np * (max_val - min_val) + min_val

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    elif out_type == np.uint16:
        img_np = (img_np * scale_factor * 256.0).round()
    elif out_type == np.float32:
        pass  # already in float32
    else:
        raise ValueError(f'Unsupported output type: {out_type}')

    return img_np.astype(out_type).squeeze()

##Changed Here! Fixed scaling artifacts by using consistent processing for all tiles
def postprocess(images, out_type=np.uint8, scale_factor=1, min_val=None, max_val=None, scaleFactor=None):
    """
    Post-process model outputs with consistent scaling to prevent tile edge artifacts.
    Uses uniform processing for all tiles to ensure proper averaging during reconstruction.
    """
    if scaleFactor is not None:
        if isinstance(scaleFactor, list):
            # Convert list of tensors to floats and invert (e.g., 0.5 -> 2.0)
            # Use first element to ensure consistent scaling across all tiles
            first_val = scaleFactor[0]
            if isinstance(first_val, torch.Tensor):
                scaleFactor = 1.0 / float(first_val.item())
            else:
                scaleFactor = 1.0 / float(first_val)
        elif isinstance(scaleFactor, tuple):
            # Use first element to ensure consistent scaling across all tiles
            scaleFactor = 1.0 / float(scaleFactor[0])
        elif isinstance(scaleFactor, torch.Tensor):
            # Use first element or single value to ensure consistent scaling
            if scaleFactor.numel() == 1:
                scaleFactor = 1.0 / float(scaleFactor.item())
            else:
                scaleFactor = 1.0 / float(scaleFactor.flatten()[0].item())
        else:
            # Handle numeric types
            scaleFactor = 1.0 / float(scaleFactor)

        # Apply SAME scaleFactor to ALL tiles to maintain consistency
        return [
            tensor2img(
                F.interpolate(image.unsqueeze(0), scale_factor=scaleFactor, mode='bicubic', align_corners=False).squeeze(0),
                out_type=out_type,
                scale_factor=scale_factor,
                min_val=min_val,
                max_val=max_val
            )
            for image in images
        ]
    else:
        # Apply SAME processing to ALL tiles to maintain consistency
        return [
            tensor2img(
                image,
                out_type=out_type,
                scale_factor=scale_factor,
                min_val=min_val,
                max_val=max_val
            )
            for image in images
        ]


def set_seed(seed, gl_seed=0):
	"""  set random seed, gl_seed used in worker_init_fn function """
	if seed >=0 and gl_seed>=0:
		seed += gl_seed
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)

	''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
		speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
	if seed >=0 and gl_seed>=0:  # slower, more reproducible
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	else:  # faster, less reproducible
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True

def set_gpu(args, distributed=False, rank=0):
	""" set parameter to gpu or ddp """
	if args is None:
		return None
	if distributed and isinstance(args, torch.nn.Module):
		return DDP(args.cuda(), device_ids=[rank], output_device=rank, broadcast_buffers=True, find_unused_parameters=True)
	else:
		return args.cuda()
		
def set_device(args, distributed=False, rank=0):
	""" set parameter to gpu or cpu """
	if torch.cuda.is_available():
		if isinstance(args, list):
			return (set_gpu(item, distributed, rank) for item in args)
		elif isinstance(args, dict):
			return {key:set_gpu(args[key], distributed, rank) for key in args}
		else:
			args = set_gpu(args, distributed, rank)
	return args



