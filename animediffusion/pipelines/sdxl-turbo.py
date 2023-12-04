import logging

import torch
from diffusers import StableDiffusionXLPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LatentConsistencyModelStableXLTxt2ImgPipeline:

    def __init__(self, stable_model_id):
        self.pipeline = None
        self.device = None
        self.set_device()

        if self.pipeline is None:
            self.load_model(stable_model_id)

    def set_device(self):
        """Sets the device to be used for inference based on availability."""
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Using device: {self.device}")

    def load_model(self, stable_model_id: str):
        logging.info('Loading Latent Consistency Model (SDXL)')
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path=stable_model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

        self.pipeline.to(self.device)
