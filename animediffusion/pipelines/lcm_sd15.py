import logging
from typing import List, Optional, Union

import torch
from diffusers import LatentConsistencyModelPipeline

from ..utils.scheduler_list import get_scheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LCM_SD15:

    def __init__(self, stable_model_id, scheduler: Optional[str] = None):
        self.pipeline = None
        self.device = None
        self.set_device()

        if self.pipeline is None:
            self.load_model(stable_model_id, scheduler)

    def set_device(self):
        """Sets the device to be used for inference based on availability."""
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Using device: {self.device}")

    def load_model(self, stable_model_id: str, scheduler: Optional[str] = None):
        logging.info('Loading Latent Consistency Model (SD15)')
        self.pipeline = LatentConsistencyModelPipeline.from_pretrained(
            pretrained_model_name_or_path=stable_model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        if scheduler is not None:
            self.pipeline.scheduler = get_scheduler(self.pipeline, scheduler)

        self.pipeline.to(self.device)

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        original_inference_steps: int = None,
        timesteps: List[int] = None,
        guidance_scale: float = 8.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
    ):

        results = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            original_inference_steps=original_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            prompt_embeds=prompt_embeds,
            clip_skip=clip_skip,
        )

        return results
