import logging

import torch
from diffusers import ControlNetModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class StableDiffusionXLControlnetPipeline:

    def __init__(self, stable_model_id, controlnet_model_path, task_id):
        self.task_id = task_id
        self.pipeline = None
        self.controlnet = None
        self.device = None
        self.set_device()

        if self.pipeline is None:
            self.load_model(stable_model_id, controlnet_model_path)

    def set_device(self):
        """Sets the device to be used for inference based on availability."""
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Using device: {self.device}")

    def load_model(self, stable_model_id: str, controlnet_model_path: str):
        if self.task_id == "txt2img":
            from diffusers import StableDiffusionXLControlNetPipeline

            logging.info("StableXLText2ImgPipeline: Loading Model")

            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_model_path,
                torch_dtype=torch.float16,
            )

            self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                pretrained_model_name_or_path=stable_model_id,
                torch_dtype=torch.float16,
                controlet=self.controlnet,
                variant="fp16",
                use_safetensors=True,
            )

        elif self.task_id == "img2img":
            from diffusers import StableDiffusionXLControlNetImg2ImgPipeline

            logging.info("StableXLImg2ImgPipeline: Loading Model")

            self.pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                pretrained_model_name_or_path=stable_model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
        elif self.task_id == "img2inpaint":
            from diffusers import StableDiffusionXLControlNetInpaintPipeline

            logging.info("StableXLImg2InpaintPipeline: Loading Model")

            self.pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                pretrained_model_name_or_path=stable_model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )

        self.pipeline.to(self.device)
