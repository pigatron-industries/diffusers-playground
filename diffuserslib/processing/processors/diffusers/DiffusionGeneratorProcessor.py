from ..ImageProcessor import ImageProcessor, ImageContext
from ....inference import DiffusersPipelines, GenerationParameters, ModelParameters
from typing import Callable, Dict, Any, List


class DiffusionGeneratorProcessor(ImageProcessor):
    def __init__(self, 
                 pipelines:DiffusersPipelines|Callable[[], DiffusersPipelines], 
                 prompt:str|Callable[[], str]="black and white gradient", 
                 model:str|Callable[[], str]="runwayml/stable-diffusion-v1-5"):
        args = {
            "pipelines": pipelines,
            "prompt": prompt,
            "model": model
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        pipelines = args["pipelines"]
        initImage = inputImages[0].getViewportImage()

        params = GenerationParameters(
            safetychecker = False,
            prompt = args["prompt"],
            negprompt = "",
            steps = 20,
            width = initImage.size[0],
            height = initImage.size[1],
            scheduler = "DPMSolverMultistepScheduler",
            models = [ ModelParameters(name = args["model"]) ]
        )

        image, _ = pipelines.generate(params)
        outputImage.setViewportImage(image)
        return outputImage
    