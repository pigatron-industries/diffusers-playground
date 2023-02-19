from .. import evaluateArguments
from PIL import Image, ImageDraw


class GeometryTask():
    pass


class GraphicsContext():
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


class GeometryPipeline():
    def __init__(self, size=(512, 512), background="white"):
        self.initargs = {
            "size": size,
            "background": background
        }
        self.tasks = []

    def addTask(self, task):
        self.tasks.append(task)

    def __call__(self):
        initargs = evaluateArguments(self.initargs)
        context = GraphicsContext(size=initargs["size"])
        for task in self.tasks:
            task(context)
        background = Image.new("RGBA", size=context.size, color=initargs["background"])
        background.alpha_composite(context.getViewportImage(), (0, 0))
        return background
