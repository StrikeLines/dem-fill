import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp
from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric
from datetime import datetime
import os
import rasterio
import numpy as np
import math
from generate_tile_worldfiles import generate_worldfiles_for_tiles

torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy('file_system')


def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and and cuDNN environment '''
    # Enable cuDNN with memory-efficient settings
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Configure PyTorch for memory efficiency
    torch.backends.cuda.max_split_size_mb = 512  # Limit CUDA memory splits
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for better performance
    
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    # Configure parallel processing for test phase
    if opt['phase'] == 'test':
        # Optimize dataloader settings for parallel inference
        opt['datasets'][opt['phase']]['dataloader']['args'].update({
            'batch_size': opt['batch'] if opt['batch'] is not None else 4,  # Use command line batch size or default to 4
            'num_workers': min(8, os.cpu_count()),  # Reduce worker count
            'pin_memory': True,  # Speed up CPU to GPU transfers
            'prefetch_factor': 2,  # Reduce prefetch to save memory
            'persistent_workers': True  # Keep workers alive between batches
        })
        
        # Enable memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)  # Leave some GPU memory free
    
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    try:
        if opt['phase'] == 'train':
            model.train()
        else:
            model.test()
    finally:
        phase_writer.close()
        

def create_temp_dir() -> str:
    """Create a temporary directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = f"temp/tiling_{timestamp}"
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def cleanup_temp_dir(temp_dir: str) -> None:
    """Remove temporary directory and all contents"""
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


def create_nodata_mask(input_file, output_file=None, mask_value_nodata=255, mask_value_data=0):
    """
    Create a nodata mask from a TIF file.
    
    Args:
        input_file (str): Path to input TIF file
        output_file (str): Path to output mask file (optional)
        mask_value_nodata (int): Value to assign to nodata pixels in mask (default: 255)
        mask_value_data (int): Value to assign to valid data pixels in mask (default: 0)
    
    Returns:
        str: Path to the created mask file
    """
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_nodata_mask.tif"
    
    print(f"Creating nodata mask for: {input_file}")
    print(f"Output mask will be saved as: {output_file}")
    
    # Open the input file
    with rasterio.open(input_file) as src:
        print(f"Input file info:")
        print(f"  - Dimensions: {src.width} x {src.height}")
        print(f"  - Bands: {src.count}")
        print(f"  - Data type: {src.dtypes[0]}")
        print(f"  - NoData value: {src.nodata}")
        
        # Read all bands (though we'll process band by band)
        masks = []
        
        for band_idx in range(1, src.count + 1):
            print(f"Processing band {band_idx}...")
            
            # Read the band data
            data = src.read(band_idx)
            
            # Create the nodata mask
            if src.nodata is not None:
                # Use the defined nodata value
                nodata_mask = (data == src.nodata)
                print(f"  - Using defined nodata value: {src.nodata}")
            else:
                # If no nodata value is defined, try to detect common patterns
                print("  - No nodata value defined, attempting to detect...")
                
                # Common nodata patterns to check
                potential_nodata = [
                    data == -99999.0,
                    data == -9999.0,
                    data == -999.0,
                    np.isnan(data),
                    np.isinf(data)
                ]
                
                # Combine all potential nodata masks
                nodata_mask = np.logical_or.reduce(potential_nodata)
                
                # Also check for extreme outliers (very large or very small values)
                if data.dtype in [np.float32, np.float64]:
                    data_finite = data[np.isfinite(data)]
                    if len(data_finite) > 0:
                        q1, q99 = np.percentile(data_finite, [1, 99])
                        data_range = q99 - q1
                        outlier_threshold = 100 * data_range  # 100x the data range
                        
                        extreme_outliers = (
                            (data < (q1 - outlier_threshold)) |
                            (data > (q99 + outlier_threshold))
                        )
                        nodata_mask = nodata_mask | extreme_outliers
            
            # Convert boolean mask to integer values
            mask = np.where(nodata_mask, mask_value_nodata, mask_value_data).astype(np.uint8)
            masks.append(mask)
            
            # Print statistics
            nodata_count = np.sum(nodata_mask)
            valid_count = np.sum(~nodata_mask)
            total_pixels = mask.size
            nodata_percentage = (nodata_count / total_pixels) * 100
            
            print(f"  - Total pixels: {total_pixels:,}")
            print(f"  - NoData pixels: {nodata_count:,} ({nodata_percentage:.2f}%)")
            print(f"  - Valid data pixels: {valid_count:,} ({100-nodata_percentage:.2f}%)")
        
        # Prepare the profile for the output file
        profile = src.profile.copy()
        profile.update({
            'dtype': 'uint8',
            'nodata': None,  # Mask files typically don't have nodata values
            'compress': 'lzw'  # Add compression to save space
        })
        
        # Write the mask file
        print(f"\nWriting mask to: {output_file}")
        with rasterio.open(output_file, 'w', **profile) as dst:
            for band_idx, mask in enumerate(masks, 1):
                dst.write(mask, band_idx)
        
        print(f"Nodata mask created successfully!")
        print(f"Mask file saved as: {output_file}")
        
        return output_file

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/dem_completion.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='test')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('--distributed', action='store_true', help='Enable distributed processing')
    parser.add_argument('-o', '--output_dir_name', default=None)
    parser.add_argument('-pt', '--preprocess_type', default=None)
    parser.add_argument('-rs', '--resume_state', default=None)
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--mask_root', default=None)
    parser.add_argument('--use_color_map', type=bool, default=False)
    parser.add_argument('--out_type', type=str, default='float32')
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--n_timestep', type=int, default=None)
    parser.add_argument('--sample_num', type=int, default=0)
    parser.add_argument('--input_img', default=None, help='Path to input GeoTIFF image')
    parser.add_argument('--input_mask', default=None, help='Path to input mask image (optional - will be auto-generated if not provided)')
    parser.add_argument('--tile_size', default=128, help='Tile size in pixels defualts to 128')
    parser.add_argument('--tile_overlap', default=0, help='Pixel overlap in tiles')
    parser.add_argument('--keep_temp', action='store_true', help='Keep temporary files after processing')
    parser.add_argument('--nodata_value', default=None, help='NoData value in the input image')
    

    ''' parser configs '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    
    # print(opt)
    # exit()
    # Check if single image input mode
    tiler = None
    mask_tiler = None
    if opt.get('input_img'):
        print("=" * 60)
        print("GEOTIFF TILING WORKFLOW")
        print("=" * 60)
        
        # Create temporary directory
        temp_dir = create_temp_dir()
        tiles_dir = os.path.join(temp_dir, "tiles")
        masks_dir = os.path.join(temp_dir, "masks")
        print(f"Created temporary directory: {temp_dir}")
        
        # Check if mask is provided, if not create one automatically
        if not opt.get('input_mask') or not os.path.exists(opt.get('input_mask', '')):
            print("\n" + "─" * 40)
            print("PREPROCESSING: CREATING NODATA MASK")
            print("─" * 40)
            
            # Generate mask filename in temp directory
            input_basename = os.path.basename(opt['input_img'])
            mask_filename = os.path.splitext(input_basename)[0] + "_auto_nodata_mask.tif"
            auto_mask_path = os.path.join(temp_dir, mask_filename)
            
            try:
                # Create the nodata mask automatically
                created_mask_path = create_nodata_mask(
                    opt['input_img'],
                    auto_mask_path,
                    mask_value_nodata=255,  # Standard mask value for nodata
                    mask_value_data=0       # Standard mask value for valid data
                )
                
                # Set the mask path for the workflow
                opt['input_mask'] = created_mask_path
                print(f"✓ Auto-generated mask will be used: {created_mask_path}")
                
            except Exception as e:
                print(f"Error creating automatic nodata mask: {str(e)}")
                print("Proceeding without mask - will generate from nodata values during tiling")
                opt['input_mask'] = None
        else:
            print(f"Using provided mask: {opt['input_mask']}")
        
        # Initialize tilers
        tile_overlap = opt.get('tile_overlap',0)
        tile_size = opt.get('tile_size',128)
        try:
            if not opt.get('input_mask') or not os.path.exists(opt.get('input_mask', '')):
                tiler = Util.GeoTiffTiler(tile_size=tile_size, overlap=tile_overlap)
                
                # Tile the input image and auto-generate mask from nodata
                print("\n" + "─" * 40)
                print("Tiling input image and generating mask from nodata")
                print("─" * 40)
                img_metadata, mask_metadata = tiler.tile_image_with_mask(opt['input_img'],
                            tiles_dir,
                            masks_dir,
                            prefix_img = "tile",
                            prefix_mask = "mask",
                            nodata_value = opt.get('nodata_value',None))

            else:
                tiler = Util.GeoTiffTiler(tile_size=tile_size, overlap=tile_overlap)
                mask_tiler = Util.GeoTiffTiler(tile_size=tile_size, overlap=tile_overlap)

                # Tile the input image
                print("\n" + "─" * 40)
                print("Tiling input image")
                print("─" * 40)

                img_metadata = tiler.tile_image(opt['input_img'], tiles_dir, prefix="dem_tile")

                # Tile the mask image
                print("\n" + "─" * 40)
                print("Tiling mask image")
                print("─" * 40)
                mask_metadata = mask_tiler.tile_image(opt['input_mask'], masks_dir, prefix="mask_tile")


            # # Save image metadata and create filelist
            # img_metadata_path = os.path.join(tiles_dir, "metadata.json")
            # tiler.save_metadata(img_metadata, img_metadata_path)
            # metadata_path = img_metadata_path  # Store for later use
            
            # Create image filelist
            img_flist_path = tiler.makeflist(img_metadata, tiles_dir)
            opt["datasets"][opt['phase']]["which_dataset"]["args"]["data_root"] = img_flist_path
            print(f"Image tiles filelist created: {img_flist_path}")
            
            # # Save mask metadata and create filelist
            # mask_metadata_path = os.path.join(masks_dir, "metadata.json")
            # mask_tiler.save_metadata(mask_metadata, mask_metadata_path)
            
            # Create mask filelist
            mask_flist_path = tiler.makeflist(mask_metadata, masks_dir)
            opt["datasets"][opt['phase']]["which_dataset"]["args"]["mask_root"] = mask_flist_path

            print(f"Mask tiles filelist created: {mask_flist_path}")
            
            print("\n" + "─" * 40)
            print("Ready for model inference")
            print("─" * 40)
            
        except Exception as e:
            print(f"Error during tiling: {str(e)}")
            if temp_dir and os.path.exists(temp_dir):
                cleanup_temp_dir(temp_dir)
            raise
        finally:
            # Clean up mask tiler (we only need the main tiler for merging)
            del mask_tiler
                
    
    # Generate world files for predicted output directory (before inference starts)
    if opt.get('input_img') and tiler:
        print("\n" + "─" * 40)
        print("GENERATING WORLD FILES FOR INFERENCE OUTPUT")
        print("─" * 40)
        
        # Predict the output tiles directory path
        output_tiles_dir = os.path.join(opt['path']['results'], 'test', '0')
        
        # Create the directory if it doesn't exist
        os.makedirs(output_tiles_dir, exist_ok=True)
        
        tile_overlap = opt.get('tile_overlap', 0)
        tile_size = opt.get('tile_size', 128)
        
        try:
            # Generate world files based on predicted tile layout from metadata
            from generate_tile_worldfiles import calculate_tile_worldfile_params, write_world_file
            
            # Get the image metadata to predict tile layout
            with rasterio.open(opt['input_img']) as src:
                original_transform = src.transform
                width, height = src.width, src.height
            
            # Calculate tiling parameters
            step_size = tile_size - tile_overlap
            tiles_x = math.ceil((width - tile_overlap) / step_size)
            tiles_y = math.ceil((height - tile_overlap) / step_size)
            
            print(f"Creating world files for predicted {tiles_x}x{tiles_y} = {tiles_x * tiles_y} output tiles...")
            
            worldfiles_created = 0
            for row in range(tiles_y):
                for col in range(tiles_x):
                    # Create predicted tile filename
                    tile_filename = f"dem_tile_{row:04d}_{col:04d}.tif"
                    world_filename = f"dem_tile_{row:04d}_{col:04d}.tfw"
                    world_path = os.path.join(output_tiles_dir, world_filename)
                    
                    # Calculate world file parameters for this tile
                    params = calculate_tile_worldfile_params(
                        original_transform, tile_size, row, col, step_size
                    )
                    
                    # Write world file
                    write_world_file(world_path, params)
                    worldfiles_created += 1
                    
                    if worldfiles_created % 100 == 0:
                        print(f"Generated {worldfiles_created} world files...")
            
            print(f"✓ {worldfiles_created} world files pre-generated for output directory: {output_tiles_dir}")
            
        except Exception as e:
            print(f"Warning: Could not generate world files for output directory: {str(e)}")
            print("Proceeding with inference - world files can be generated manually later")
    
    '''cuda devices'''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1 
        main_worker(0, 1, opt)
        
    
    # Post-processing: Merge tiles back if tiled
    if tiler and temp_dir:
        print("\n" + "=" * 60)
        print("POST-PROCESSING: MERGING TILES")
        print("=" * 60)
        
        try:
            # Load metadata and merge tiles
            print("Loading metadata and merging tiles...")
            
            # img_metadata = tiler.load_metadata(metadata_path)
            
            # print(opt)
            # Determine output path
            if opt.get('path') and opt.get('path').get('results'):
                # Use results directory from config for working files
                results_dir = opt.get('path').get('results')
                os.makedirs(results_dir, exist_ok=True)
                
                # Create output directory in the root of the repo for final stitched file
                output_dir = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "output")
                os.makedirs(output_dir, exist_ok=True)
                
                output_filename = os.path.basename(opt['input_img'])
                name, ext = os.path.splitext(output_filename)
                base_output = f"{name}_processed{ext}"
                
                # Check if file exists and generate new name if needed
                counter = 1
                output_filename = base_output
                while os.path.exists(os.path.join(output_dir, output_filename)):
                    output_filename = f"{name}_processed-{counter}{ext}"
                    counter += 1
                
                # Save the final stitched file to the output directory
                output_image = os.path.join(output_dir, output_filename)
                
                # Merge tiles back to single georeferenced image
                tiles_dir = os.path.join(results_dir, r'test/0')
                tiler.merge_tiles(tiles_dir, img_metadata, output_image, crop_to_original=True)
                print(f"\n✓ Final merged image saved to: {output_image}")
            else:
                # Default output path
                print(f"Error during tile merging")
                print("Temporary files preserved for debugging")
            

            
        except Exception as e:
            raise e
            print(f"Error during tile merging: {str(e)}")
            print("Temporary files preserved for debugging")
        finally:
            # Clean up temporary files (unless user wants to keep them)
            if not opt.get('keep_temp', False):
                print("\n" + "─" * 40)
                print("CLEANUP: Removing temporary files")
                print("─" * 40)
                cleanup_temp_dir(temp_dir)
            else:
                print(f"\nTemporary files preserved at: {temp_dir}")
