import torch
import torch.nn as nn
from typing import Dict, Union, Optional, Callable, Tuple
from contextlib import contextmanager

def rescale(x, in_range, out_range, clamp = False):
    old_min, old_max = in_range
    new_min, new_max = out_range

    x = (x - old_min) / (old_max - old_min)
    x = x * (new_max - new_min) + new_min

    if clamp:
        x = x.clamp(new_min, new_max)
    
    return x

def preprocess_input_image(image_tensor):
    # Rescale from [0, 255] to [-1, 1]
    image_tensor = rescale(image_tensor, (0, 255), (-1, 1))
    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    image_tensor = image_tensor.unsqueeze(0)
    # Permute dimensions: (1, H, W, C) -> (1, C, H, W)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    
    return image_tensor

def postprocess_output_image(images):
    # Rescale from [-1, 1] to [0, 255] and clamp
    images = rescale(images, (-1, 1), (0, 255), clamp=True)
    # Permute dimensions: (B, C, H, W) -> (B, H, W, C)
    images = images.permute(0, 2, 3, 1)
    # Convert to uint8 and move to CPU as numpy array
    images = images.to(torch.uint8).cpu().numpy()
    
    return images

def get_timestep_embedding(timestep, freqs = None):
    if freqs is None:
        freqs = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32) / 160)

    t = torch.tensor([timestep], dtype=torch.float32).unsqueeze(1)   # shape (1,1)
    f = freqs.unsqueeze(0)                                          # shape (1,160)
    x = t * f                                                      # shape (1,160)

    emb = torch.cat([torch.cos(x), torch.sin(x)], dim=1)            # shape (1,320)

    return emb



@contextmanager
def capture_activations(
    module: nn.Module,
    *,
    where: Optional[Callable[[str, nn.Module], bool]] = None,
    leaves_only: bool = True,
    clone: bool = True,
    detach: bool = True,
    cpu: bool = True,
):
    activations: Dict[str, Union[torch.Tensor, Tuple]] = {}
    handles = []

    def _prep(t):
        if t is None:
            return None
        if detach and hasattr(t, "requires_grad") and t.requires_grad:
            t = t.detach()
        if clone:
            t = t.clone()
        if cpu:
            t = t.cpu()
        return t

    try:
        for name, subm in module.named_modules():
            if name == "":
                continue  # skip root
            if leaves_only and any(subm.children()):
                continue
            if where is not None and not where(name, subm):
                continue

            def make_hook(nm: str):
                def _hook(mod, inp, out):
                    if torch.is_tensor(out):
                        activations[nm] = _prep(out)
                    elif isinstance(out, (list, tuple)):
                        activations[nm] = tuple(_prep(o) if torch.is_tensor(o) else o for o in out)
                    elif isinstance(out, dict):
                        activations[nm] = {k: _prep(v) if torch.is_tensor(v) else v for k, v in out.items()}
                    else:
                        activations[nm] = out
                return _hook

            handles.append(subm.register_forward_hook(make_hook(name)))

        yield activations
    finally:
        for h in handles:
            h.remove()