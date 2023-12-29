from ..ImageProcessor import ImageProcessor, ImageContext
from ....inference import DiffusersPipelines, GenerationParameters, ModelParameters, ControlImageParameters
from typing import Callable, Dict, Any, List


class DiffusionGeneratorProcessor(ImageProcessor):
    def __init__(self, 
                 pipelines:DiffusersPipelines|Callable[[], DiffusersPipelines], 
                 prompt:str|Callable[[], str]="black and white gradient", 
                 model:str|Callable[[], str]="runwayml/stable-diffusion-v1-5",
                 controlimages:List[ControlImageParameters] = []
                 ):
        args = {
            "prompt": prompt,
            "model": model
        }
        self.pipelines = pipelines
        self.controlimages = controlimages
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        pipelines = args["pipelines"]
        initImage = inputImages[0].getViewportImage()

        controlimageparams = []
        for i, controlimage in enumerate(self.controlimages):
            controlimageparams.append(
                ControlImageParameters(
                    image = inputImages[i].getViewportImage(),
                    type = controlimage.type,
                    model = controlimage.model,
                )
            )

        params = GenerationParameters(
            safetychecker = False,
            prompt = args["prompt"],
            negprompt = "",
            steps = 20,
            width = initImage.size[0],
            height = initImage.size[1],
            scheduler = "DPMSolverMultistepScheduler",
            models = [ ModelParameters(name = args["model"]) ],
            controlimages = []
        )

        image, _ = pipelines.generate(params)
        outputImage.setViewportImage(image)
        return outputImage
    