import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from .diffusers_pipeline import DEMInpaintingPipeline
class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule
        
        # Diffusers pipeline for fast inference (lazy initialization)
        self._diffusers_pipeline = None
        self._use_diffusers = False

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    def enable_diffusers(self, scheduler_type="dpmpp"):
        """
        Enable fast diffusers-based inference
        
        Args:
            scheduler_type: One of 'dpmpp', 'unipc', 'ddim'
        """
        self._use_diffusers = True
        if self._diffusers_pipeline is None:
            self._diffusers_pipeline = DEMInpaintingPipeline.from_network(
                self, scheduler_type=scheduler_type
            )
        else:
            self._diffusers_pipeline.set_scheduler(scheduler_type)
    
    def disable_diffusers(self):
        """Disable diffusers and use original DDPM sampling"""
        self._use_diffusers = False
    
    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, num_inference_steps=None):
        """
        Unified restoration method supporting both original DDPM and fast diffusers
        
        Args:
            y_cond: Conditioning input
            y_t: Initial noise (optional)
            y_0: Ground truth for inpainting (optional)
            mask: Binary mask (optional)
            sample_num: Number of intermediate samples to return (original method)
            num_inference_steps: Number of steps for diffusers (if None, auto-select)
        
        Returns:
            (final_result, intermediate_results) for compatibility
        """
        
        if self._use_diffusers and self._diffusers_pipeline is not None:
            # Use fast diffusers sampling
            if num_inference_steps is None:
                # Auto-select reasonable number of steps based on scheduler
                scheduler_name = self._diffusers_pipeline.scheduler.__class__.__name__
                if "DPM" in scheduler_name or "UniPC" in scheduler_name:
                    num_inference_steps = 20  # Very fast
                else:
                    num_inference_steps = 50  # Still much faster than original
            
            print(f"Using {self._diffusers_pipeline.scheduler.__class__.__name__} with {num_inference_steps} steps")
            
            # Run diffusers inference
            if sample_num > 0:
                result, intermediates = self._diffusers_pipeline(
                    y_cond=y_cond,
                    y_t=y_t,
                    y_0=y_0,
                    mask=mask,
                    num_inference_steps=num_inference_steps,
                    return_intermediate=True,
                    sample_num=sample_num
                )
                return result, intermediates
            else:
                result = self._diffusers_pipeline(
                    y_cond=y_cond,
                    y_t=y_t,
                    y_0=y_0,
                    mask=mask,
                    num_inference_steps=num_inference_steps,
                    return_intermediate=False
                )
                return result, result
        
        else:
            # Use original DDPM sampling
            print(f"Using original DDPM sampling with {self.num_timesteps} steps")
            return self._restoration_original(y_cond, y_t, y_0, mask, sample_num)
    
    @torch.no_grad()
    def _restoration_original(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        """Original DDPM restoration method (preserved for comparison)"""
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = None if sample_num == 0 else (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            if mask is not None:
                y_t = y_0*(1.-mask) + mask*y_t
            if sample_inter is not None and i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr
    
    def benchmark_inference(self, y_cond, mask=None, y_0=None):
        """
        Benchmark different inference methods
        
        Returns:
            Dictionary with timing results
        """
        import time
        results = {}
        
        # Test original method
        start_time = time.time()
        original_result, _ = self._restoration_original(y_cond, mask=mask, y_0=y_0, sample_num=0)
        original_time = time.time() - start_time
        results['original_ddpm'] = {
            'time': original_time,
            'steps': self.num_timesteps,
            'result': original_result
        }
        
        # Test diffusers methods
        for scheduler_type in ['dpmpp', 'unipc', 'ddim']:
            for steps in [10, 20, 50]:
                self.enable_diffusers(scheduler_type)
                
                start_time = time.time()
                diffusers_result, _ = self.restoration(
                    y_cond, mask=mask, y_0=y_0, sample_num=0, num_inference_steps=steps
                )
                diffusers_time = time.time() - start_time
                
                results[f'{scheduler_type}_{steps}'] = {
                    'time': diffusers_time,
                    'steps': steps,
                    'speedup': original_time / diffusers_time,
                    'result': diffusers_result
                }
        
        return results

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        if mask is not None:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)

            # from torchinfo import summary
            # params = {}#unet.copy()
            # params['input_size'] = (4,2,256,256)
            # params['gammas'] = sample_gammas
            # summary(self.denoise_fn, **params, col_names=("input_size", "output_size", "num_params"), depth=10)

            if hasattr(self.loss_fn, '__name__') and self.loss_fn.__name__ in ['masked_mse_loss', 'masked_l1_loss']:
                loss = self.loss_fn(mask*noise, mask*noise_hat, mask) # might not be necessary 
            else:
                loss = self.loss_fn(mask*noise, mask*noise_hat)
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            loss = self.loss_fn(noise, noise_hat)
        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


