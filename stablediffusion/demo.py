import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import pipeline

DEVICE = "cpu"

tokenizer = CLIPTokenizer("data/vocab.json", "data/merges.txt")

model_file_path  = "data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_stardard_weights(model_file_path, DEVICE)

# text to image
prompt = "A cat in the style of Monet."
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 7

sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt = prompt,
    uncond_prompt = uncond_prompt,
    do_cfg = do_cfg,
    cfg_scale = cfg_scale,
    sampler_name = sampler,
    n_inference_steps = num_inference_steps,
    seed = seed,
    models = models,
    device = DEVICE,
    idle_device = DEVICE,
    tokenizer = tokenizer
)

Image.fromarray(output_image).save("output.png")