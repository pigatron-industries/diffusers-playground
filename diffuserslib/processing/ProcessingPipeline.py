from PIL import Image
from ..batch import evaluateArguments, PlaceholderArgument


class ImageProcessor():
    pass


class ImageContext():
    def __init__(self, size, oversize=256):
        self.oversize = oversize
        self.size = size
        self.offset = (oversize, oversize)
        if(size is not None):
            self.viewport = (oversize, oversize, oversize+size[0], oversize+size[1])
            self.image = Image.new("RGBA", size=(size[0]+oversize*2, size[1]+oversize*2))
        else:
            self.viewport = (oversize, oversize, oversize, oversize)
            self.image = None

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
    def __init__(self, size=None, oversize=256):
        self.initargs = {
            "size": size
        }
        self.oversize = oversize
        self.tasks = []
        self.context = None

    def hasPlaceholder(self, name):
        # check if placeholder argument exists in initargs or tasks
        for key, arg in self.initargs.items():
            if isinstance(arg, PlaceholderArgument) and arg.name == name:
                return True
        for task in self.tasks:
            if hasattr(task, "args"):
                for key, arg in task.args.items():
                    if isinstance(arg, PlaceholderArgument) and arg.name == name:
                        return True
        return False
    
    def getPlaceholderValues(self, name):
        # find placeholder argument in initargs and tasks and return value
        placeholders = []
        for key, arg in self.initargs.items():
            if (isinstance(arg, PlaceholderArgument) and arg.name == name):
                placeholders.append(arg.getValue())
        for task in self.tasks:
            for key, arg in task.args.items():
                if (isinstance(arg, PlaceholderArgument) and arg.name == name):
                    placeholders.append(arg.getValue())
        return placeholders

    def setPlaceholder(self, name, value):
        # find placeholder argument in initargs and tasks and set value
        for key, arg in self.initargs.items():
            if (isinstance(arg, PlaceholderArgument) and arg.name == name):
                self.initargs[key].setValue(value)
        for task in self.tasks:
            for key, arg in task.args.items():
                if (isinstance(arg, PlaceholderArgument) and arg.name == name):
                    task.args[key].setValue(value)

    def addTask(self, task):
        self.tasks.append(task)

    def __call__(self):
        initargs = evaluateArguments(self.initargs)
        self.context = ImageContext(size=initargs["size"], oversize=self.oversize)
        for task in self.tasks:
            task(self.context)
        return self.context.getViewportImage()
    
    def getLastOutput(self):
        return self.context.getViewportImage()

