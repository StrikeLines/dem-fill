import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        
        # Filter out DDIM-specific parameters before passing to make_beta_schedule
        beta_schedule_params = {k: v for k, v in self.beta_schedule[phase].items()
                               if k not in ['ddim_steps', 'ddim_eta']}
        betas = make_beta_schedule(**beta_schedule_params)
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

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
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

    @torch.no_grad()
    def ddim_restoration(self, y_cond, y_t=None, y_0=None, mask=None,
                         ddim_steps=50, eta=0.0, sample_num=8):
        """DDIM sampling that closely follows original DDPM structure"""
        b, *_ = y_cond.shape

        # Create DDIM timestep schedule that matches original training
        assert self.num_timesteps % ddim_steps == 0 or ddim_steps >= self.num_timesteps, \
            f"ddim_steps ({ddim_steps}) must divide num_timesteps ({self.num_timesteps}) or be >= num_timesteps"
        
        if ddim_steps >= self.num_timesteps:
            # Use all timesteps (same as DDPM)
            timesteps = list(range(self.num_timesteps))[::-1]
            ddim_eta = 1.0  # Force stochastic sampling when using all steps
        else:
            # Subsample timesteps uniformly
            skip = self.num_timesteps // ddim_steps
            timesteps = list(range(self.num_timesteps - 1, -1, -skip))
            if timesteps[-1] != 0:
                timesteps.append(0)
            ddim_eta = eta

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = None if sample_num == 0 else (len(timesteps) // sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((b,), t, device=y_cond.device, dtype=torch.long)
            
            if ddim_steps >= self.num_timesteps:
                # Use original DDPM sampling (p_sample)
                y_t = self.p_sample(y_t, t_tensor, y_cond=y_cond)
            else:
                # Use DDIM sampling
                y_t = self.ddim_p_sample(y_t, t_tensor, timesteps, i, ddim_eta, y_cond=y_cond)
            
            # Apply mask for inpainting
            if mask is not None:
                y_t = y_0*(1.-mask) + mask*y_t
                
            # Store intermediates for visualization
            if sample_inter is not None and i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
    
        return y_t, ret_arr

    @torch.no_grad()
    def ddim_p_sample(self, y_t, t, timesteps, i, eta, y_cond=None):
        """DDIM version of p_sample that uses existing p_mean_variance"""
        # Use existing p_mean_variance to get x0 prediction and posterior mean
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=True, y_cond=y_cond)
        
        # Get x0 prediction (reverse the posterior calculation)
        # model_mean = posterior_mean_coef1 * x0 + posterior_mean_coef2 * y_t
        # Solve for x0: x0 = (model_mean - posterior_mean_coef2 * y_t) / posterior_mean_coef1
        posterior_mean_coef1 = extract(self.posterior_mean_coef1, t, y_t.shape)
        posterior_mean_coef2 = extract(self.posterior_mean_coef2, t, y_t.shape)
        x0_pred = (model_mean - posterior_mean_coef2 * y_t) / posterior_mean_coef1
        
        # Get next timestep
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            t_next_tensor = torch.full(t.shape, t_next, device=t.device, dtype=t.dtype)
            
            # DDIM step
            alpha_t = extract(self.gammas, t, x_shape=(1, 1, 1, 1))
            alpha_next = extract(self.gammas, t_next_tensor, x_shape=(1, 1, 1, 1))
            
            # Compute DDIM variance
            sigma_t = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
            
            # Compute predicted noise
            eps = (y_t - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(1 - alpha_t)
            
            # DDIM update
            pred_dir = torch.sqrt(1 - alpha_next - sigma_t**2) * eps
            noise = torch.randn_like(y_t) if eta > 0 else torch.zeros_like(y_t)
            y_next = torch.sqrt(alpha_next) * x0_pred + pred_dir + sigma_t * noise
            
            return y_next
        else:
            # Final step - return x0 prediction
            return x0_pred

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


