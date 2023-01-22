from .. import evaluateArguments
from PIL import Image, ImageDraw


class GeometryTask():
    pass


class GraphicsContext():
    def __init__(self, size, oversize=256):
        self.image = Image.new("RGBA", size=(size[0]+oversize*2, size[1]+oversize*2))
        self.viewport = (oversize, oversize, oversize+size[0], oversize+size[1])
        self.offset = (oversize, oversize)
        self.size = size

    def getViewportImage(self):
        return self.image.crop(self.viewport)


class GeometryPipeline():
    def __init__(self, size=(512, 512), colour="white"):
        self.initargs = {
            "size": size,
            "colour": colour
        }
        self.tasks = []

    def addTask(self, task):
        self.tasks.append(task)

    def __call__(self):
        initargs = evaluateArguments(self.initargs)
        context = GraphicsContext(size=initargs["size"])
        for task in self.tasks:
            task(context)
        background = Image.new("RGBA", size=initargs["size"], color=initargs["colour"])
        background.alpha_composite(context.getViewportImage(), (0, 0))
        return background
