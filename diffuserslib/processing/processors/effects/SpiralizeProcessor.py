from ..ImageProcessor import ImageProcessor, ImageContext
from PIL import Image
from typing import Dict, Any, List


class SpiralizeProcessor(ImageProcessor):
    def __init__(self, rotation=180, steps=4, zoom=2):
        args = {
            "rotation": rotation,
            "steps": steps,
            "zoom": zoom
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        angle = args["rotation"] / args["steps"]
        zoom = ((args["zoom"]-1) / args["steps"]) + 1
        modimages = []
        modimage = inputImages[0].getFullImage()
        for i in range(args["steps"]):
            modimage = modimage.rotate(angle)
            modimage = self.zoomOut(modimage, zoom)
            modimages.append(modimage)
        image = inputImages[0].getFullImage()
        for modimage in modimages:
            image.alpha_composite(modimage, (0, 0))
        outputImage.setFullImage(image)
        return outputImage

    def zoomOut(self, image, ratio):
        modimage = Image.new('RGBA', (int(image.width*ratio), int(image.height*ratio)), (255, 255, 255, 0))
        xpos = (modimage.width - image.width) // 2
        ypos = (modimage.height - image.height) // 2
        modimage.paste(image, (xpos, ypos))
        return modimage.resize((image.width, image.height), resample=Image.BICUBIC)
        