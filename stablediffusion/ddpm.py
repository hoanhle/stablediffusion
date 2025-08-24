import torch
import numpy as np

class DDPMSampler:
    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        # linear-in-sqrt(beta) schedule (SD v1 style)
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5,
                                    num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.one = torch.tensor(1.0, dtype=torch.float32)
        self.generator = generator

        self.num_training_steps = num_training_steps
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(
            np.arange(0, num_training_steps)[::-1].copy()
        ).to(torch.long)

    def set_inference_timesteps(self, num_inference_steps: int = 50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        ts = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)
        self.timesteps = torch.from_numpy(ts).to(torch.long)

    def _get_prev_timestep(self, timestep: int) -> int:
        return timestep - (self.num_training_steps // self.num_inference_steps)

    @torch.no_grad()
    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        # move ᾱ to correct device/dtype for math & indexing
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device, dtype=torch.long)

        sqrt_alphas_cumprod = (alphas_cumprod[timesteps]) ** 0.5
        while len(sqrt_alphas_cumprod.shape) < len(original_samples.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)

        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(original_samples.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)

        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )
        noisy_samples = sqrt_alphas_cumprod * original_samples + sqrt_one_minus_alphas_cumprod * noise
        return noisy_samples

    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_timestep = self._get_prev_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.one.to(alpha_prod_t)
        current_beta_t = 1.0 - alpha_prod_t / alpha_prod_t_prev

        # Eq. (7)
        variance = ((1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)
        return variance

    def set_strength(self, strength: float = 1.0):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.num_inference_steps = int(self.timesteps.numel())

    @torch.no_grad()
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        t = int(timestep)
        prev_t = self._get_prev_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one.to(alpha_prod_t)

        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t

        # predicted x0 (Eq. 15)
        pred_original_sample = (latents - (beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5)

        # μ_t coefficients (Eq. 7)
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5) * current_beta_t / beta_prod_t
        current_sample_coeff = (current_alpha_t ** 0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        if t > 0:
            variance = self._get_variance(t).to(device=model_output.device, dtype=model_output.dtype)
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            pred_prev_sample = pred_prev_sample + (variance ** 0.5) * noise

        return pred_prev_sample
