from PIL import Image
from ..batch import evaluateArguments, PlaceholderArgument
from typing import Optional, Tuple, Callable, Self, List
from .processors import ImageContext, ImageProcessor


class ImageProcessorPipeline():
    def __init__(self, size:Tuple[int, int]|Callable[[], Tuple[int, int]]|None = None, oversize:int = 256):
        self.initargs = {
            "size": size
        }
        self.oversize = oversize
        self.tasks:List[ImageProcessor] = []
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
        for i, inputIndex in enumerate(task.getInputIndexes()):
            if (inputIndex < 0):
                task.getInputIndexes()[i] = len(self.tasks) + inputIndex
        self.tasks.append(task)
        return self


    def __call__(self):
        self.initcontext = self.getInitImage()
        for task in self.tasks:
            inputImages = self.getInputImages(task)
            task(inputImages)
        return self.getLastOutput()


    def getInitImage(self) -> ImageContext:
        args = evaluateArguments(self.initargs)
        return ImageContext(size=args["size"], oversize=self.oversize)


    def getInputImages(self, task:ImageProcessor) -> List[ImageContext]:
        inputImages = []
        for inputIndex in task.getInputIndexes():
            if(inputIndex >= 0):
                outputImage = self.tasks[inputIndex].getOutputImage()
                inputImages.append(outputImage)
            else:
                inputImages.append(self.initcontext)
        return inputImages


    def getLastOutput(self) -> ImageContext:
        return self.tasks[-1].getOutputImage()

