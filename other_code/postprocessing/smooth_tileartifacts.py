import rasterio
import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter
import os
from pathlib import Path

class TileArtifactSmoother:
    def __init__(self, tile_size=128, overlap=64, kernel_size=3, boundary_width=3):
        """
        Initialize the tile artifact smoother
        
        Args:
            tile_size: Size of tiles (128x128)
            overlap: Overlap between tiles (64 pixels)
            kernel_size: Size of kernel for mean smoothing (keep small, 3 is good)
            boundary_width: Width of boundary region to smooth (3 pixels = center + 1 on each side)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.step_size = tile_size - overlap  # 128 - 64 = 64
        self.kernel_size = kernel_size
        self.boundary_width = boundary_width
    
    def create_tile_boundary_mask(self, height, width, detected_boundaries_x, detected_boundaries_y, mask_data=None):
        """
        Create a mask for tile boundaries with specified width
        
        Args:
            height, width: Image dimensions
            detected_boundaries_x, detected_boundaries_y: Lists of detected boundary positions
            mask_data: Original mask array, only smooth where mask > 0
            
        Returns:
            Boolean mask where True indicates areas to smooth
        """
        boundary_mask = np.zeros((height, width), dtype=bool)
        
        # Create boundaries with specified width around detected positions
        half_width = self.boundary_width // 2
        
        # Vertical boundaries
        for x in detected_boundaries_x:
            x_start = max(0, x - half_width)
            x_end = min(width, x + half_width + 1)
            boundary_mask[:, x_start:x_end] = True
        
        # Horizontal boundaries  
        for y in detected_boundaries_y:
            y_start = max(0, y - half_width)
            y_end = min(height, y + half_width + 1)
            boundary_mask[y_start:y_end, :] = True
        
        # Only keep boundaries where there's actual filled data
        if mask_data is not None:
            filled_areas = mask_data > 0
            boundary_mask = boundary_mask & filled_areas
            
        return boundary_mask
    
    def detect_actual_tile_boundaries(self, data, mask_data=None, nodata_value=None):
        """
        Detect actual tile boundaries by looking for discontinuities in the data
        """
        print("Analyzing data to detect tile boundaries...")
        
        # Create valid data mask
        if nodata_value is not None and not np.isnan(nodata_value):
            valid_mask = data != nodata_value
        elif np.isnan(nodata_value) if nodata_value is not None else False:
            valid_mask = ~np.isnan(data)
        else:
            valid_mask = np.ones_like(data, dtype=bool)
        
        # If we have mask data, only consider filled areas
        if mask_data is not None:
            valid_mask = valid_mask & (mask_data > 0)
        
        height, width = data.shape
        
        print(f"Valid data pixels: {np.sum(valid_mask):,}")
        
        # Check theoretical boundaries first
        theoretical_boundaries_x = list(range(self.step_size, width, self.step_size))
        theoretical_boundaries_y = list(range(self.step_size, height, self.step_size))
        
        print(f"Checking {len(theoretical_boundaries_x)} vertical and {len(theoretical_boundaries_y)} horizontal boundaries...")
        
        # Look for actual discontinuities near theoretical boundaries
        detected_x = []
        detected_y = []
        
        # Check vertical boundaries
        for x in theoretical_boundaries_x:
            if x >= width - 1:
                continue
                
            search_range = range(max(1, x-2), min(width-1, x+3))
            best_x = None
            max_discontinuity = 0
            
            for test_x in search_range:
                if test_x >= width-1:
                    continue
                    
                # Calculate discontinuity at this column
                left_col = data[:, test_x-1]
                right_col = data[:, test_x+1]
                
                # Only consider valid pixels
                valid_left = valid_mask[:, test_x-1]
                valid_right = valid_mask[:, test_x+1]
                both_valid = valid_left & valid_right
                
                if np.sum(both_valid) > height // 10:  # Need some data
                    discontinuity = np.mean(np.abs(left_col[both_valid] - right_col[both_valid]))
                    if discontinuity > max_discontinuity and discontinuity > 0.001:  # Minimum threshold
                        max_discontinuity = discontinuity
                        best_x = test_x
            
            if best_x is not None:
                detected_x.append(best_x)
        
        # Check horizontal boundaries
        for y in theoretical_boundaries_y:
            if y >= height - 1:
                continue
                
            search_range = range(max(1, y-2), min(height-1, y+3))
            best_y = None
            max_discontinuity = 0
            
            for test_y in search_range:
                if test_y >= height-1:
                    continue
                    
                # Calculate discontinuity at this row
                top_row = data[test_y-1, :]
                bottom_row = data[test_y+1, :]
                
                # Only consider valid pixels
                valid_top = valid_mask[test_y-1, :]
                valid_bottom = valid_mask[test_y+1, :]
                both_valid = valid_top & valid_bottom
                
                if np.sum(both_valid) > width // 10:  # Need some data
                    discontinuity = np.mean(np.abs(top_row[both_valid] - bottom_row[both_valid]))
                    if discontinuity > max_discontinuity and discontinuity > 0.001:  # Minimum threshold
                        max_discontinuity = discontinuity
                        best_y = test_y
            
            if best_y is not None:
                detected_y.append(best_y)
        
        print(f"Detected {len(detected_x)} vertical boundaries: {detected_x[:10]}{'...' if len(detected_x) > 10 else ''}")
        print(f"Detected {len(detected_y)} horizontal boundaries: {detected_y[:10]}{'...' if len(detected_y) > 10 else ''}")
        
        # Create boundary mask with proper width
        boundary_mask = self.create_tile_boundary_mask(height, width, detected_x, detected_y, mask_data)
        
        print(f"Total boundary pixels: {np.sum(boundary_mask):,}")
        return boundary_mask
    
    def apply_gentle_smoothing(self, data, mask, kernel_size, nodata_value=None):
        """
        Apply gentle smoothing with reasonable thresholds
        """
        result = data.copy()
        original_dtype = data.dtype
        
        # Create valid mask
        if nodata_value is not None and not np.isnan(nodata_value):
            valid_mask = data != nodata_value
        elif np.isnan(nodata_value) if nodata_value is not None else False:
            valid_mask = ~np.isnan(data)
        else:
            valid_mask = np.ones_like(data, dtype=bool)
        
        # Only smooth where we have valid data and want to smooth
        smooth_region = mask & valid_mask
        
        if np.sum(smooth_region) == 0:
            print("No pixels to smooth!")
            return result
        
        print(f"Smoothing {np.sum(smooth_region):,} pixels...")
        
        # Calculate global statistics for reasonable thresholds
        valid_data = data[valid_mask]
        data_std = np.std(valid_data)
        max_reasonable_change = data_std * 0.5  # Allow changes up to 0.5 standard deviations
        
        print(f"Data std: {data_std:.6f}, max allowed change: {max_reasonable_change:.6f}")
        
        # Apply scipy uniform filter first (it's much faster)
        work_data = data.astype(np.float64)
        
        # Create a mask for the filter - set invalid areas to mean value so they don't affect smoothing
        filter_data = work_data.copy()
        mean_val = np.mean(valid_data)
        filter_data[~valid_mask] = mean_val
        
        # Apply uniform filter
        smoothed_data = uniform_filter(filter_data, size=kernel_size, mode='reflect')
        
        # Only apply changes where reasonable and requested
        smooth_indices = np.where(smooth_region)
        changes_applied = 0
        changes_skipped = 0
        
        for i in range(len(smooth_indices[0])):
            row, col = smooth_indices[0][i], smooth_indices[1][i]
            original_value = work_data[row, col]
            new_value = smoothed_data[row, col]
            change = abs(new_value - original_value)
            
            # Apply more lenient threshold
            if change <= max_reasonable_change:
                result[row, col] = new_value.astype(original_dtype)
                changes_applied += 1
            else:
                changes_skipped += 1
                # Apply partial change (50% of the smoothed change)
                partial_change = original_value + 0.5 * (new_value - original_value)
                result[row, col] = partial_change.astype(original_dtype)
                changes_applied += 1
        
        print(f"Applied {changes_applied:,} changes, skipped {changes_skipped:,} extreme changes")
        return result
    
    def save_debug_mask(self, mask, output_path, suffix='debug_mask'):
        """Save debug mask to see what's being smoothed"""
        debug_path = output_path.replace('.tif', f'_{suffix}.tif')
        
        # Create a simple profile for the debug mask
        profile = {
            'driver': 'GTiff',
            'height': mask.shape[0],
            'width': mask.shape[1],
            'count': 1,
            'dtype': 'uint8',
            'compress': 'lzw'
        }
        
        with rasterio.open(debug_path, 'w', **profile) as dst:
            dst.write((mask * 255).astype('uint8'), 1)
        
        print(f"Debug mask saved to: {debug_path}")

    def smooth_raster(self, input_path, mask_path=None, output_path=None):
        """
        Main function to smooth tile artifacts with improved handling
        """
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_smoothed{input_path_obj.suffix}")
        
        print(f"Processing: {input_path}")
        print(f"Output: {output_path}")
        print(f"Boundary width: {self.boundary_width} pixels")
        
        # Read input raster
        with rasterio.open(input_path) as src:
            data = src.read()
            profile = src.profile.copy()
            nodata_value = src.nodata
            
        print(f"Input shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"NoData value: {nodata_value}")
        
        # Analyze data range
        for band_idx in range(data.shape[0]):
            band_data = data[band_idx]
            
            if nodata_value is not None and not np.isnan(nodata_value):
                valid_data = band_data[band_data != nodata_value]
            elif np.isnan(nodata_value) if nodata_value is not None else False:
                valid_data = band_data[~np.isnan(band_data)]
            else:
                valid_data = band_data.flatten()
            
            if len(valid_data) > 0:
                print(f"Band {band_idx+1} - Valid pixels: {len(valid_data):,}")
                print(f"Band {band_idx+1} - Range: {np.min(valid_data):.6f} to {np.max(valid_data):.6f}")
                print(f"Band {band_idx+1} - Mean: {np.mean(valid_data):.6f}, Std: {np.std(valid_data):.6f}")
        
        # Process each band separately
        smoothed_data = data.copy()
        
        for band_idx in range(data.shape[0]):
            band_data = data[band_idx]
            print(f"\n=== Processing band {band_idx + 1} ===")
            
            # Read mask data if provided
            mask_data = None
            if mask_path is not None:
                try:
                    with rasterio.open(mask_path) as mask_src:
                        mask_data = mask_src.read(1)
                        
                    # Ensure mask has same dimensions
                    if mask_data.shape != band_data.shape:
                        print(f"Resizing mask from {mask_data.shape} to {band_data.shape}")
                        from scipy.ndimage import zoom
                        zoom_factors = (band_data.shape[0] / mask_data.shape[0],
                                      band_data.shape[1] / mask_data.shape[1])
                        mask_data = zoom(mask_data, zoom_factors, order=0)
                        
                    print(f"Mask loaded: {np.sum(mask_data > 0):,} filled pixels")
                    
                except Exception as e:
                    print(f"Warning: Could not load mask: {e}")
                    mask_data = None
            
            # Detect actual tile boundaries
            boundary_mask = self.detect_actual_tile_boundaries(
                band_data, 
                mask_data, 
                nodata_value
            )
            
            if np.sum(boundary_mask) == 0:
                print("No tile boundaries detected - skipping smoothing")
                continue
            
            # # Save debug mask
            # if band_idx == 0:
            #     self.save_debug_mask(boundary_mask, output_path, 'boundaries_to_smooth')
            
            # Apply gentle smoothing
            print("Applying smoothing...")
            smoothed_band = self.apply_gentle_smoothing(
                band_data, 
                boundary_mask, 
                self.kernel_size,
                nodata_value
            )
            
            smoothed_data[band_idx] = smoothed_band
            
            # Verify results
            changes = smoothed_band != band_data
            if np.sum(changes) > 0:
                print(f"Modified {np.sum(changes):,} pixels")
                
                # Check ranges properly
                if nodata_value is not None and not np.isnan(nodata_value):
                    valid_original = band_data[band_data != nodata_value]
                    valid_smoothed = smoothed_band[smoothed_band != nodata_value]
                else:
                    valid_original = band_data.flatten()
                    valid_smoothed = smoothed_band.flatten()
                
                print(f"Original range: {np.min(valid_original):.6f} to {np.max(valid_original):.6f}")
                print(f"Smoothed range: {np.min(valid_smoothed):.6f} to {np.max(valid_smoothed):.6f}")
                
                # Calculate actual change statistics
                changed_pixels = (band_data != smoothed_band) & (band_data != nodata_value if nodata_value is not None else True)
                if np.sum(changed_pixels) > 0:
                    changes_array = smoothed_band[changed_pixels] - band_data[changed_pixels]
                    print(f"Change statistics - Mean: {np.mean(changes_array):.6f}, Std: {np.std(changes_array):.6f}")
                    print(f"Change range: {np.min(changes_array):.6f} to {np.max(changes_array):.6f}")
            else:
                print("No pixels were modified")
        
        # Save result
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(smoothed_data)
        
        print(f"\nSmoothed raster saved to: {output_path}")
        return output_path

if __name__ == "__main__":
    smoother = TileArtifactSmoother(
        tile_size=128,
        overlap=64,
        kernel_size=9,        # larger kernel for better smoothing
        boundary_width=7      # 7-pixel wide boundary pixels to be smoothened
    )
    
    smoother.smooth_raster(
        input_path="Path to result from the DiffDem Process", 
        mask_path=r"Path to mask used", 
        output_path=r"Path for output postprocessed image"
    )