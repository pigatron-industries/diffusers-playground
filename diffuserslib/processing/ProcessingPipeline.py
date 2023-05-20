from PIL import Image
from ..batch import evaluateArguments, PlaceholderArgument
from typing import Optional, Tuple, Callable, Self


class ImageContext():
    def __init__(self, size:Tuple[int, int]|None, oversize:int=256):
        self.oversize = oversize
        self.size = size
        self.offset = (oversize, oversize)
        if(size is not None):
            self.viewport = (oversize, oversize, oversize+size[0], oversize+size[1])
            self.image = Image.new("RGBA", size=(size[0]+oversize*2, size[1]+oversize*2))
        else:
            self.viewport = (oversize, oversize, oversize, oversize)
            self.image = None

    def setViewportImage(self, image:Image.Image):
        self.image = Image.new("RGBA", size=(image.width+self.oversize*2, image.height+self.oversize*2))
        self.image.paste(image, self.offset)
        self.calcSize()

    def getViewportImage(self) -> Optional[Image.Image]:
        if(self.image is not None):
            return self.image.crop(self.viewport)

    def calcSize(self):
        if(self.image is not None):
            self.size = (self.image.width-self.oversize*2, self.image.height-self.oversize*2)
            self.viewport = (self.oversize, self.oversize, self.oversize+self.size[0], self.oversize+self.size[1])


class ImageProcessor():
    def __init__(self):
        self.args = {}

    def __call__(self, context:ImageContext):
        raise NotImplementedError
    
    def evaluateArguments(self, context:ImageContext):
        return evaluateArguments(self.args, context=context)


class ImageProcessorPipeline():
    def __init__(self, size:Tuple[int, int]|Callable[[], Tuple[int, int]]|None = None, oversize:int = 256):
        self.initargs = {
            "size": size
        }
        self.oversize = oversize
        self.tasks = []
        self.context = None

    def hasPlaceholder(self, name:str) -> bool:
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
                arg.setValue(value)
        for task in self.tasks:
            for key, arg in task.args.items():
                if (isinstance(arg, PlaceholderArgument) and arg.name == name):
                    arg.setValue(value)

    def addTask(self, task) -> Self:
        self.tasks.append(task)
        return self

    def __call__(self):
        initargs = evaluateArguments(self.initargs)
        self.context = ImageContext(size=initargs["size"], oversize=self.oversize)
        for task in self.tasks:
            task(self.context)
        return self.context.getViewportImage()
    
    def getLastOutput(self) -> Optional[Image.Image]:
        if (self.context is not None):
            return self.context.getViewportImage()

