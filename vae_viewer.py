import numpy as np
import torch
from PIL import Image
from imgui_bundle import imgui
import pyviewer_extended
from pyviewer.params import *  # type: ignore
from stablediffusion.model_loader import preload_models_from_stardard_weights
from stablediffusion.utils import preprocess_input_image, postprocess_output_image

@strict_dataclass
class State(ParamContainer):
    seed: Param = IntParam('Seed', 0, 0, 1000)
    image_path: Param = StringParam('Image Path', 'images/cat-dog.jpeg')
    ui_scale: Param = FloatParam('UI Scale', 5.0, 0.1, 5.0)

class VAEViewer(pyviewer_extended.MultiTexturesDockingViewer):
    def setup_state(self):
        self.state = State()
        self.models = preload_models_from_stardard_weights("data/v1-5-pruned-emaonly.ckpt", "cpu")
        self.encoder = self.models["encoder"]
        self.decoder = self.models["decoder"]
        
        self.original_image = None
        self.reconstructed_image = None
        self.force_recompute = False

    def load_image(self, image_path, size=(512, 512)):
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(size)
            return np.array(img)
        except Exception as e:
            raise e
   
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
            latents = self.encoder(input_tensor, noise)
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


        return {
            'Original': self.original_image.astype(np.float32) / 255.0,
            'Reconstructed': self.reconstructed_image.astype(np.float32) / 255.0,
        }

    @pyviewer_extended.dockable
    def toolbar(self):
        imgui.text("VAE Viewer")
        _, self.state.image_path = imgui.input_text("Image Path", self.state.image_path, 256)
        if imgui.button("Load Image"):
            self.force_recompute = True
        

if __name__ == '__main__':
    _ = VAEViewer(
        'VAE Viewer', 
        ['Original', 'Reconstructed'], 
        enable_vsync=True
    )
