from .ImageProcessor import ImageProcessor
from ...batch import evaluateArguments

from PIL import Image, ImageOps


class SymmetrizeProcessor(ImageProcessor):
    """ horizontal, vertical, or rotation """
    def __init__(self, type="horizontal"):
        self.args = {
            "type": type
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        if(args["type"] == "horizontal"):
            modimage = ImageOps.flip(context.image)
        elif(args["type"] == "vertical"):
            modimage =  ImageOps.mirror(context.image)
        elif(args["type"] == "rotation"):
            modimage =  context.image.rotate(180)
        else:
            return context
        context.image.alpha_composite(modimage, (0, 0))
        return context


class SimpleTransformProcessor(ImageProcessor):
    """ 
        type = none, fliphorizontal, flipvertical, rotate90, rotate180, rotate270 
        rotate90 and rotate270 also swap viewport dimensions
    """
    def __init__(self, type="fliphorizontal"):
        self.args = {
            "type": type
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        if(args["type"] == "flipvertical"):
            modimage = ImageOps.flip(context.image)
        elif(args["type"] == "fliphorizontal"):
            modimage = ImageOps.mirror(context.image)
        elif(args["type"] == "rotate180"):
            modimage = context.image.transpose(Image.ROTATE_180)
        elif(args["type"] == "rotate90"):
            modimage = context.image.transpose(Image.ROTATE_90)
        elif(args["type"] == "rotate270"):
            modimage = context.image.transpose(Image.ROTATE_270)
        else:
            return context
        context.image = modimage
        context.calcSize()
        return context


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
        