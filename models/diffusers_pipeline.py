import torch
import numpy as np
from typing import Optional, Union, Dict, Any
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, UniPCMultistepScheduler, DDPMScheduler
from diffusers.schedulers import DDIMScheduler
from tqdm import tqdm

def extract(a, t, x_shape=(1,1,1,1)):
    """
    CRITICAL FIX: This is the exact extract function from the original network.py
    We must use the same noise level calculation format!
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DEMInpaintingPipeline(DiffusionPipeline):
    """
    Custom Diffusers pipeline for DEM inpainting using advanced samplers.
    
    This pipeline wraps the existing U-Net architecture and provides
    fast inference using DPM-Solver++, UniPC, DDIM, and other advanced schedulers.
    """
    
    def __init__(self, unet, scheduler=None, torch_dtype=torch.float32):
        super().__init__()
        
        self.unet = unet
        self.torch_dtype = torch_dtype
        
        # Default to DPM-Solver++ for speed
        if scheduler is None:
            scheduler = DPMSolverMultistepScheduler.from_config({
                "num_train_timesteps": 1000,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "beta_schedule": "linear",
                "solver_order": 2,
                "prediction_type": "epsilon",
            })
        
        self.scheduler = scheduler
        
        # Store original gammas for compatibility
        if hasattr(unet, 'gammas'):
            self.register_buffer('gammas', unet.gammas)
        
    @classmethod
    def from_network(cls, network, scheduler_type="dpmpp", **kwargs):
        """
        Create pipeline from existing Network object
        
        Args:
            network: The existing Network object with denoise_fn
            scheduler_type: Type of scheduler ('dpmpp', 'unipc', 'ddim', 'ddpm')
        """
        
        # REVERT DESTRUCTIVE CHANGES: Use diffusers-friendly beta values
        # The original config beta_end=0.09 is too high for diffusers schedulers!
        
        # Use beta values that work well with modern diffusers schedulers
        beta_start = 1e-4  # This is fine
        beta_end = 0.02    # REVERT: Use diffusers-friendly value, not 0.09!
        num_timesteps = network.num_timesteps  # Use actual network timesteps
        
        print(f"[DIFFUSERS] Using diffusers-compatible beta schedule: start={beta_start}, end={beta_end}, steps={num_timesteps}")
        
        # Map scheduler types to configurations
        scheduler_configs = {
            "dpmpp": {
                "class": DPMSolverMultistepScheduler,
                "config": {
                    "num_train_timesteps": num_timesteps,
                    "beta_start": beta_start,
                    "beta_end": beta_end,
                    "beta_schedule": "linear",
                    "solver_order": 2,
                    "prediction_type": "epsilon",
                }
            },
            "unipc": {
                "class": UniPCMultistepScheduler,
                "config": {
                    "num_train_timesteps": num_timesteps,
                    "beta_start": beta_start,
                    "beta_end": beta_end,
                    "beta_schedule": "linear",
                    "prediction_type": "epsilon",
                }
            },
            "ddim": {
                "class": DDIMScheduler,
                "config": {
                    "num_train_timesteps": num_timesteps,
                    "beta_start": beta_start,
                    "beta_end": beta_end,
                    "beta_schedule": "linear",
                    "prediction_type": "epsilon",
                }
            },
            "ddpm": {
                "class": DDPMScheduler,
                "config": {
                    "num_train_timesteps": num_timesteps,
                    "beta_start": beta_start,
                    "beta_end": beta_end,
                    "beta_schedule": "linear",
                    "prediction_type": "epsilon",
                }
            }
        }
        
        if scheduler_type not in scheduler_configs:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}. Choose from: {list(scheduler_configs.keys())}")
        
        scheduler_info = scheduler_configs[scheduler_type]
        scheduler = scheduler_info["class"].from_config(scheduler_info["config"])
        
        # Store network reference for potential future use
        pipeline = cls(network.denoise_fn, scheduler, **kwargs)
        pipeline._original_network = network
        
        return pipeline
    
    def set_scheduler(self, scheduler_type: str):
        """Change scheduler type on the fly"""
        # REVERT: Use diffusers-compatible beta values
        num_timesteps = getattr(self._original_network, 'num_timesteps', 1000) if hasattr(self, '_original_network') else 1000
        beta_start = 1e-4
        beta_end = 0.02  # Diffusers-friendly value
        
        if scheduler_type == "dpmpp":
            self.scheduler = DPMSolverMultistepScheduler.from_config({
                "num_train_timesteps": num_timesteps,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "beta_schedule": "linear",
                "solver_order": 2,
                "prediction_type": "epsilon",
            })
        elif scheduler_type == "unipc":
            self.scheduler = UniPCMultistepScheduler.from_config({
                "num_train_timesteps": num_timesteps,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "beta_schedule": "linear",
                "prediction_type": "epsilon",
            })
        elif scheduler_type == "ddim":
            self.scheduler = DDIMScheduler.from_config({
                "num_train_timesteps": num_timesteps,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "beta_schedule": "linear",
                "prediction_type": "epsilon",
            })
        elif scheduler_type == "ddpm":
            self.scheduler = DDPMScheduler.from_config({
                "num_train_timesteps": num_timesteps,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "beta_schedule": "linear",
                "prediction_type": "epsilon",
            })
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        print(f"[DIFFUSERS] Updated scheduler {scheduler_type} with beta: start={beta_start}, end={beta_end}, steps={num_timesteps}")
    
    @torch.no_grad()
    def __call__(
        self,
        y_cond: torch.Tensor,
        y_0: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        y_t: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        return_intermediate: bool = False,
        sample_num: int = 0,
        **kwargs
    ):
        """
        Fast inference using advanced schedulers
        
        Args:
            y_cond: Conditioning input (e.g., partial DEM + mask)
            y_0: Ground truth for inpainting (optional)
            mask: Binary mask indicating areas to inpaint (1=inpaint, 0=keep)
            y_t: Initial noise (optional, will generate if None)
            num_inference_steps: Number of denoising steps (20-50 typical)
            guidance_scale: Guidance scale (for future classifier-free guidance)
            eta: Amount of noise to add for DDIM (0.0 = deterministic)
            generator: Random number generator
            return_intermediate: Whether to return intermediate steps
            sample_num: Legacy parameter for compatibility
        
        Returns:
            Generated DEM tensor, optionally with intermediate steps
        """
        
        device = y_cond.device
        batch_size = y_cond.shape[0]
        
        # Set up the scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Initialize noise
        if y_t is None:
            if generator is not None:
                y_t = torch.randn(y_cond.shape, generator=generator, device=device, dtype=self.torch_dtype)
            else:
                y_t = torch.randn_like(y_cond, device=device, dtype=self.torch_dtype)
        
        # Collect intermediate results if requested
        intermediates = []
        if return_intermediate and sample_num > 0:
            sample_inter = max(1, num_inference_steps // sample_num)
        else:
            sample_inter = None
        
        # Denoising loop using advanced scheduler
        progress_bar = tqdm(timesteps, desc=f'Inference ({self.scheduler.__class__.__name__})')
        
        for i, t in enumerate(progress_bar):
            # Expand timestep for batch
            timestep = t.expand(batch_size)
            
            # REVERT: Use scheduler's noise level calculation (compatible with diffusers)
            # The original gamma calculation doesn't work well with diffusers schedulers
            if hasattr(self.scheduler, 'alphas_cumprod'):
                # Calculate noise level from scheduler's alpha values
                alpha_cumprod_t = self.scheduler.alphas_cumprod[t]
                noise_level = (1 - alpha_cumprod_t).sqrt().expand(batch_size, 1).to(device)
            else:
                # Fallback for schedulers without alphas_cumprod
                noise_level = torch.sqrt(t.float() / self.scheduler.num_train_timesteps).expand(batch_size, 1).to(device)
            
            # Predict noise using U-Net
            model_input = torch.cat([y_cond, y_t], dim=1)
            noise_pred = self.unet(model_input, noise_level)
            
            # Scheduler step
            if isinstance(self.scheduler, DDIMScheduler):
                scheduler_output = self.scheduler.step(noise_pred, t, y_t, eta=eta, generator=generator)
            elif isinstance(self.scheduler, UniPCMultistepScheduler):
                # UniPC doesn't support generator argument
                scheduler_output = self.scheduler.step(noise_pred, t, y_t)
            elif isinstance(self.scheduler, DDPMScheduler):
                # DDPM supports generator
                scheduler_output = self.scheduler.step(noise_pred, t, y_t, generator=generator)
            else:
                # DPM-Solver++ and others support generator
                scheduler_output = self.scheduler.step(noise_pred, t, y_t, generator=generator)
            y_t = scheduler_output.prev_sample
            
            # Apply mask if provided (inpainting)
            if mask is not None and y_0 is not None:
                # Keep original values where mask=0, inpaint where mask=1
                y_t = y_0 * (1. - mask) + mask * y_t
            
            # Store intermediate results
            if sample_inter is not None and (i + 1) % sample_inter == 0:
                intermediates.append(y_t.clone())
        
        # Return results
        if return_intermediate and intermediates:
            return y_t, torch.stack(intermediates, dim=0)
        else:
            return y_t
    
    def benchmark_schedulers(
        self, 
        y_cond: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        y_0: Optional[torch.Tensor] = None,
        steps_list: list = [10, 20, 30, 50]
    ):
        """
        Benchmark different schedulers and step counts
        
        Returns:
            Dictionary with timing and quality metrics
        """
        import time
        
        schedulers = ["dpmpp", "unipc", "ddim"]
        results = {}
        
        original_scheduler = self.scheduler
        
        for scheduler_name in schedulers:
            results[scheduler_name] = {}
            self.set_scheduler(scheduler_name)
            
            for steps in steps_list:
                start_time = time.time()
                
                # Run inference
                result = self(
                    y_cond=y_cond,
                    y_0=y_0,
                    mask=mask,
                    num_inference_steps=steps
                )
                
                end_time = time.time()
                
                results[scheduler_name][steps] = {
                    "time": end_time - start_time,
                    "result": result
                }
        
        # Restore original scheduler
        self.scheduler = original_scheduler
        
        return results