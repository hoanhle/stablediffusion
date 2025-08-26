import numpy as np
import torch
from PIL import Image
from imgui_bundle import imgui
import pyviewer_extended
from pyviewer.params import *  # type: ignore
from stablediffusion.model_loader import preload_models_from_stardard_weights
from stablediffusion.utils import preprocess_input_image, postprocess_output_image, capture_activations
from enum import Enum
import math

# ---------- helpers ----------
def create_encoder_layer_enum(activation_keys):
    enum_dict = {key: key for key in activation_keys}
    return Enum('VAE_EncoderLayer', enum_dict)

# simple, fast BWR colormap: x in [-1,1] → RGB in [0,1]
def bwr_map(x: np.ndarray) -> np.ndarray:
    # x assumed in [-1, 1]
    a = np.clip(x, -1.0, 1.0)
    mag = np.abs(a)                 # 0 → white, 1 → full color
    rgb = np.empty(a.shape + (3,), dtype=np.float32)
    # start from white, then subtract color from the "other" channels
    rgb[..., 0] = np.where(a >= 0, 1.0, 1.0 - mag)   # R: full for positives, fades for negatives
    rgb[..., 1] = 1.0 - mag                          # G: fades with magnitude
    rgb[..., 2] = np.where(a <= 0, 1.0, 1.0 - mag)   # B: full for negatives, fades for positives
    return rgb

def robust_symmetric_scale(x: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """Scale so that +/-q become -1/+1, center at 0. Robust to outliers."""
    q = np.nanpercentile(np.abs(x), pct)
    q = float(q) if q > 1e-12 else 1.0
    return np.clip(x / q, -1.0, 1.0).astype(np.float32)


# ---------- dummy enum (replaced later) ----------
class VAE_EncoderLayer(Enum):
    dummy = 0


@strict_dataclass
class State(ParamContainer):
    seed: Param = IntParam('Seed', 0, 0, 1000)
    image_path: Param = StringParam('Image Path', 'images/cat-dog.jpeg')
    encoder_layer: Param = EnumParam('Encoder Layer', VAE_EncoderLayer.dummy, VAE_EncoderLayer)
    heatmap_pct: Param = FloatParam('Heatmap scale (|z| percentile)', 99.5, 90.0, 100.0, 0.1)
    

class VAEViewer(pyviewer_extended.MultiTexturesDockingViewer):
    def setup_state(self):
        self.state = State()
        self.models = preload_models_from_stardard_weights("data/v1-5-pruned-emaonly.ckpt", "cpu")
        self.encoder = self.models["encoder"]
        self.decoder = self.models["decoder"]
        
        self.original_image = None
        self.reconstructed_image = None
        self.encoder_activations = {}
        self.current_activation_viz = None
        self.force_recompute = False
        self.encoder_layer_enum = None
        self.last_encoder_layer = None

    def update_encoder_layer_enum(self, activation_keys):
        if activation_keys and activation_keys != getattr(self.encoder_layer_enum, '_keys', set()):
            self.encoder_layer_enum = create_encoder_layer_enum(activation_keys)
            self.encoder_layer_enum._keys = set(activation_keys)
            
            global VAE_EncoderLayer
            VAE_EncoderLayer = self.encoder_layer_enum
            
            first_key = list(activation_keys)[0]
            self.state.encoder_layer = EnumParam('Encoder Layer', 
                                               self.encoder_layer_enum[first_key], 
                                               self.encoder_layer_enum)

    def load_image(self, image_path, size=(512, 512)):
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(size)
            return np.array(img)
        except Exception as e:
            raise e
    
    def visualize_activation(self, activation):
        if activation is None:
            return None

        # (B, C, H, W) -> (C, H, W)
        act = activation.squeeze(0)
        if act.ndim != 3:
            return None

        channels, height, width = act.shape
        grid_size = int(math.ceil(math.sqrt(channels)))
        grid_h = height * grid_size
        grid_w = width * grid_size
        grid = np.zeros((grid_h, grid_w), dtype=np.float32)

        # fill grid with z-scored channel activations (mean 0, std 1 per channel)
        act_np = act.detach().cpu().numpy()
        for i in range(channels):
            row, col = divmod(i, grid_size)
            ch = act_np[i]
            m = ch.mean()
            s = ch.std()
            if s > 1e-12:
                ch = (ch - m) / s
            else:
                ch = ch * 0.0

            r0, c0 = row * height, col * width
            grid[r0:r0 + height, c0:c0 + width] = ch

        # robust symmetric normalization so colors are comparable across layers
        grid_scaled = robust_symmetric_scale(grid, pct=float(self.state.heatmap_pct))

        # map to blue–white–red (bwr) RGB in [0,1]
        grid_rgb = bwr_map(grid_scaled)  # shape (H, W, 3), float32 in [0,1]
        return grid_rgb

    def run_vae(self, input_image):
        if input_image is None:
            return None
        
        input_tensor = preprocess_input_image(torch.tensor(input_image, dtype=torch.float32))
        
        batch_size = input_tensor.shape[0]
        latent_height = input_tensor.shape[2] // 8
        latent_width = input_tensor.shape[3] // 8
        
        torch.manual_seed(self.state.seed)
        noise = torch.randn(batch_size, 4, latent_height, latent_width)
        
        with torch.no_grad():
            with capture_activations(self.encoder) as activations:
                latents = self.encoder(input_tensor, noise)
                self.encoder_activations = activations
            
            if self.encoder_activations and not self.encoder_layer_enum:
                self.update_encoder_layer_enum(self.encoder_activations.keys())
            
            reconstructed_tensor = self.decoder(latents)
            reconstructed_image = postprocess_output_image(reconstructed_tensor)
            return reconstructed_image[0]

    def compute(self):
        if self.force_recompute or self.original_image is None:
            self.original_image = self.load_image(self.state.image_path, (512, 512))
            self.reconstructed_image = self.run_vae(self.original_image)
                
            if self.reconstructed_image is None:
                self.reconstructed_image = np.zeros((512, 512, 3), dtype=np.uint8)
                
            self.force_recompute = False

        # Get the selected layer activation and create visualization
        selected_layer = self.state.encoder_layer.value
        if selected_layer and selected_layer in self.encoder_activations and self.last_encoder_layer != selected_layer:
            print(f"Visualizing layer {selected_layer}")
            self.current_activation_viz = self.visualize_activation(self.encoder_activations[selected_layer])
            
        else:
            self.current_activation_viz = None
        
        self.last_encoder_layer = selected_layer

        # Prepare output dictionary
        output = {
            'Original': self.original_image.astype(np.float32) / 255.0,
            'Reconstructed': self.reconstructed_image.astype(np.float32) / 255.0,
        }
        
        # Add activation visualization if available
        if self.current_activation_viz is not None:
            output['Activation'] = self.current_activation_viz

        return output

    @pyviewer_extended.dockable
    def toolbar(self):
        if imgui.button("Load Image"):
            self.force_recompute = True
        
        draw_container(self.state)
        

if __name__ == '__main__':
    _ = VAEViewer(
        'VAE Viewer', 
        ['Original', 'Reconstructed', 'Activation'], 
        enable_vsync=True
    )
