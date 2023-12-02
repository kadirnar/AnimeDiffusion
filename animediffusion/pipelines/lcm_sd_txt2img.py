import logging
from typing import Optional

import torch
from diffusers import LatentConsistencyModelPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LatentConsistencyModelStableTxt2ImgPipeline:

    def __init__(self, stable_model_id):
        self.pipeline = None
        self.device = None
        self.set_device()

        if self.pipeline is None:
            self.load_model(stable_model_id=stable_model_id)

    def set_device(self):
        """Sets the device to be used for inference based on availability."""
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Using device: {self.device}")

    def load_model(self, stable_model_id: str):
        logging.info('Loading Latent Consistency Model (SD15)')

        self.pipeline = LatentConsistencyModelPipeline.from_pretrained(
            stable_model_id,
            torch_dtype=torch.float16,
        )
        self.pipeline.to(self.device)

        logging.info('Model loaded')
