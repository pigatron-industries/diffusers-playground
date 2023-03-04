from PIL import Image
from .. import evaluateArguments


class ImageProcessor():
    pass


class ImageContext():
    def __init__(self, size, oversize=256):
        self.image = Image.new("RGBA", size=(size[0]+oversize*2, size[1]+oversize*2))
        self.oversize = oversize
        self.size = size
        self.viewport = (oversize, oversize, oversize+size[0], oversize+size[1])
        self.offset = (oversize, oversize)

    def setViewportImage(self, image):
        self.image = Image.new("RGBA", size=(image.width+self.oversize*2, image.height+self.oversize*2))
        self.image.paste(image, self.offset)
        self.calcSize()

    def getViewportImage(self):
        return self.image.crop(self.viewport)

    def calcSize(self):
        self.size = (self.image.width-self.oversize*2, self.image.height-self.oversize*2)
        self.viewport = (self.oversize, self.oversize, self.oversize+self.size[0], self.oversize+self.size[1])


class ImageProcessorPipeline():
    def __init__(self, size=(512, 512)):
        self.initargs = {
            "size": size
        }
        self.tasks = []

    def addTask(self, task):
        self.tasks.append(task)

    def __call__(self):
        initargs = evaluateArguments(self.initargs)
        context = ImageContext(size=initargs["size"])
        for task in self.tasks:
            task(context)
        return context.getViewportImage()



class InitImageProcessor(ImageProcessor):
    def __init__(self, image):
        # We add all arguments to an args dictionary, 
        # some of them may be instances of Argument class which decides what the actual argument should be when it's ready to be used
        self.args = {
            "image": image
        }

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