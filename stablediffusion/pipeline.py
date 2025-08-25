import torch
import numpy as np
from tqdm import tqdm
from stablediffusion.ddpm import DDPMSampler
from stablediffusion.utils import get_timestep_embedding, preprocess_input_image, postprocess_output_image

H, W = 512, 512
LATENT_H, LATENT_W = H // 8, W // 8


def generate(prompt: str, 
             uncond_prompt: str, 
             input_image = None, 
             strength = 0.8, 
             do_cfg = True, 
             cfg_scale = 7.5, 
             sampler_name = "ddpm", 
             n_inference_steps = 50, 
             models = {},
             seed = None,
             device = None,
             idle_device = None,
             tokenizer = None):
    with torch.no_grad():
        assert 0 < strength <= 1, "strength must be between 0 and 1"
        if idle_device: 
            to_idle_device = lambda x: x.to(idle_device)
        else:
            to_idle_device = lambda x: x
        
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # convert prompt to tokens
            cond_tokens = tokenizer.batch_encode_plus([prompt], max_length=77, padding="max_length").input_ids
            # batch_size, seq_len
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # batch_size, seq_len -> batch_size, seq_len, dim
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            # 2, seq_len, dim
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus([prompt], max_length=77, padding="max_length").input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            # 1, seq_len, dim
            context = clip(tokens)
        
        to_idle_device(clip)
        
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            # TODO: add other samplers from EDM paper: https://arxiv.org/abs/2206.00364
            raise ValueError("Other samplers are not supported yet")
    
        latent_shape = (1, 4, LATENT_H, LATENT_W)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
            input_image_tensor = input_image.resize((W, H))
            input_image_tensor = torch.tensor(np.array(input_image_tensor), dtype=torch.float32, device=device)

            input_image_tensor = preprocess_input_image(input_image_tensor)

            encoder_noise = torch.randn(latent_shape, device=device, generator=generator)
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle_device(encoder)

        else:
            latents = torch.randn(latent_shape, device=device, generator=generator)
        
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        freqs = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32) / 160)
        for _, timestep in enumerate(timesteps):
            time_embedding = get_timestep_embedding(timestep, freqs).to(device)
            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)
            
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
        
            latents = sampler.step(timestep, latents, model_output)

        
        to_idle_device(diffusion)
        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle_device(decoder)

        images = postprocess_output_image(images)


        return images[0]






        