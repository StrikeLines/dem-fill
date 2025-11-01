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
    torch.backends.cudnn.enabled = True
    # warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
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

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/dem_completion.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='test')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
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
    parser.add_argument('--input_mask', default=None, help='Path to input mask image')
    parser.add_argument('--tile_size', default=128, help='Tile size in pixels defualts to 128')
    parser.add_argument('--tile_overlap', default=0, help='Pixel overlap in tiles')
    parser.add_argument('--keep_temp', action='store_true', help='Keep temporary files after processing')
    parser.add_argument('--nodata_value', default=None, help='NoData value in the input image')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM sampling steps (default: 50, faster than 1000 DDPM steps)')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta parameter (0=deterministic, 1=DDPM-like stochastic)')
    parser.add_argument('--use_ddpm', action='store_true', help='Use original DDPM sampling instead of DDIM (slower but might be more accurate)')

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
        
        # Initialize tilers
        tile_overlap = opt.get('tile_overlap',0)
        tile_size = opt.get('tile_size',128)
        try:
            if not opt.get('input_mask') or not os.path.exists(opt['input_mask']):
                tiler = Util.GeoTiffTiler(tile_size=tile_size, overlap=tile_overlap)
                
                # Tile the input image
                print("\n" + "─" * 40)
                print("Tiling input image and mask")
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
                # Use results directory from config
                results_dir = opt.get('path').get('results')
                os.makedirs(results_dir, exist_ok=True)
                output_filename = os.path.basename(opt['input_img'])
                # Add processed suffix
                name, ext = os.path.splitext(output_filename)
                output_filename = f"{name}_processed{ext}"
                output_image = os.path.join(results_dir, output_filename)
                
                # Merge tiles back to single georeferenced image
                tiles_dir = os.path.join(results_dir,r'test/0')
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
