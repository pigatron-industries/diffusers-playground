from ..ImageProcessor import ImageProcessor, ImageContext
from ....inference import DiffusersPipelines, GenerationParameters, ModelParameters
from ....batch import evaluateArguments

from PIL import ImageDraw
import math
import numpy as np
from typing import Callable


class DiffusionGeneratorProcessor(ImageProcessor):
    def __init__(self, 
                 pipelines:DiffusersPipelines|Callable[[], DiffusersPipelines], 
                 prompt:str|Callable[[], str]="black and white gradient", 
                 model:str|Callable[[], str]="runwayml/stable-diffusion-v1-5"):
        self.args = {
            "pipelines": pipelines,
            "prompt": prompt,
            "model": model
        }

    def __call__(self, context:ImageContext):
        args = evaluateArguments(self.args, context=context)
        pipelines = args["pipelines"]

        params = GenerationParameters(
            safetychecker = False,
            prompt = args["prompt"],
            negprompt = "",
            steps = 20,
            width = context.size[0],
            height = context.size[1],
            scheduler = "DPMSolverMultistepScheduler",
            models = [ ModelParameters(name = args["model"]) ]
        )

        image, _ = pipelines.generate(params)
        context.setViewportImage(image)
        return context
    