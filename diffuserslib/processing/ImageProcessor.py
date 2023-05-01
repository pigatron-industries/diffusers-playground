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
        context = ImageContext(size=initargs["size"], oversize=self.oversize)
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
    def __init__(self, type="stretch", size = (512, 768), halign="centre", valign="centre", fill="black"):
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