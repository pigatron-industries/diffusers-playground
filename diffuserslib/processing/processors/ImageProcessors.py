from PIL import Image
from .ImageProcessor import ImageProcessor
from ...batch import evaluateArguments
from typing import Union, Callable
    

class InitImageProcessor(ImageProcessor):
    def __init__(self, image):
        # We add all arguments to an args dictionary, 
        # some of them may be instances of Argument class which decides what the actual argument should be when it's ready to be used
        self.args = {
            "image": image
        }

    def setImage(self, image):
        self.args["image"] = image

    def __call__(self, context):
        # evaluateArguments decides which argument values to use at runtime
        args = evaluateArguments(self.args, context=context)
        context.setViewportImage(args["image"])
        return context
    

class FillBackgroundProcessor(ImageProcessor):
    def __init__(self, background="white"):
        self.args = {
            "background": background
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        background = Image.new("RGBA", size=context.image.size, color=args["background"])
        background.alpha_composite(context.image, (0, 0))
        context.image = background
        return context
    

class ResizeProcessor(ImageProcessor):
    def __init__(self, 
                 type:Union[str,Callable] = "stretch", 
                 size:Union[tuple[int, int],Callable] = (512, 768), 
                 halign:Union[str,Callable] = "centre", 
                 valign:Union[str,Callable] = "centre", 
                 fill:Union[str,Callable] = "black"):
        self.args = {
            "type": type,
            "size": size,
            "halign": halign,
            "valign": valign,
            "fill": fill
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        image = context.getViewportImage()
        width = args["size"][0]
        height = args["size"][1]
        newimage = image

        if(args["type"] == "stretch"):
            newimage = image.resize(args["size"], resample=Image.Resampling.LANCZOS)

        if(args["type"] == "fit"):
            ratio = min(args["size"][0]/image.width, args["size"][1]/image.height)
            image = image.resize((int(image.width*ratio), int(image.height*ratio)), resample=Image.Resampling.LANCZOS)

        if(args["type"] in ("extend", "fit")):
            newimage = Image.new("RGBA", size=(width, height), color=args["fill"])
            if (args["halign"] == "left"):
                x = 0
            elif (args["halign"] == "right"):
                x = newimage.width - image.width
            else:
                x = int((newimage.width - image.width)/2)
            if (args["valign"] == "top"):
                y = 0
            elif (args["valign"] == "bottom"):
                y = newimage.height - image.height
            else:
                y = int((newimage.height - image.height)/2)
            newimage.paste(image, (x, y))

        context.setViewportImage(newimage)
        return context
    

class CropProcessor(ImageProcessor):
    def __init__(self, size = (512, 768), position = (0.5, 0.5)):
        self.args = {
            "size": size,
            "position": position
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        image = context.getViewportImage()
        width = args["size"][0]
        height = args["size"][1]
        left = int((image.width - width) * args["position"][0])
        top = int((image.height - height) * args["position"][1])
        newimage = image.crop((left, top, left+width, top+height))
        context.setViewportImage(newimage)
        return context
