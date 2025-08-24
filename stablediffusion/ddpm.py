import torch
import numpy as np

class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0) 
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
    
    def set_inference_timesteps(self, num_inference_steps = 50):
        self.num_inference_steps = num_inference_steps
        
        step_ratio = self.num_training_steps // self.num_inference_steps
        self.timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)

    
    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alphas_cumprod = self.alpha_cumprod.to(original_samples.device, original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.float()
        while len(sqrt_alphas_cumprod.shape) < len(original_samples.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod   

        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(original_samples.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]

        noise = torch.randn_like(original_samples.shape, generator=self.generator, dtype=original_samples.dtype)
        noisy_samples = sqrt_alphas_cumprod * original_samples + sqrt_one_minus_alphas_cumprod * noise

        return noisy_samples

    def _get_prev_timestep(self, timestep: int):
        prev_timestep = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_timestep

    def _get_variance(self, timestep: int):
        prev_timestep = self._get_prev_timestep(timestep)
        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_timestep] if prev_timestep >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = current_beta_t * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)

        variance = torch.clamp(variance, min=1e-20)
        return variance


    def set_strength(self, strength = 1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.num_inference_steps = len(self.timesteps)
        
    
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_prev_timestep(t)

        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # compute preducted original sample
        pred_original_sample = (latents - (beta_prod_t ** 0.5) * model_output) / alpha_prod_t ** 0.5

        # compute the coefficent for pred_original_sample and current sample
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5) * current_beta_t / beta_prod_t 
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # compute the predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
        variance = 0

        if t > 0:
            noise = torch.randn(model_output.shape, generator=self.generator, dtype=model_output.dtype, device=model_output.device)
            variance = self._get_variance(t)
            pred_prev_sample = pred_prev_sample + (variance ** 0.5) * noise

        return pred_prev_sample
