from ..ImageProcessor import ImageProcessor
from ....batch import evaluateArguments

from PIL import Image, ImageOps


class SpiralizeProcessor(ImageProcessor):
    def __init__(self, rotation=180, steps=4, zoom=2):
        self.args = {
            "rotation": rotation,
            "steps": steps,
            "zoom": zoom
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        angle = args["rotation"] / args["steps"]
        zoom = ((args["zoom"]-1) / args["steps"]) + 1
        modimages = []
        modimage = context.image
        for i in range(args["steps"]):
            modimage = modimage.rotate(angle)
            modimage = self.zoomOut(modimage, zoom)
            modimages.append(modimage)
        for modimage in modimages:
            context.image.alpha_composite(modimage, (0, 0))
        return context

    def zoomOut(self, image, ratio):
        modimage = Image.new('RGBA', (int(image.width*ratio), int(image.height*ratio)), (255, 255, 255, 0))
        xpos = (modimage.width - image.width) // 2
        ypos = (modimage.height - image.height) // 2
        modimage.paste(image, (xpos, ypos))
        return modimage.resize((image.width, image.height), resample=Image.BICUBIC)
        