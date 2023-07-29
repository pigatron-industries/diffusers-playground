import itertools
import time
from PIL import Image
from typing import List, Dict, Callable, Tuple
import inspect
import traceback
# import pyexiv2

from IPython.display import display
import ipywidgets as widgets
import functools

from .argument.Argument import BatchArgument


def mergeDict(d1, d2):
    dict = d1
    for key in d2.keys():
        dict[key] = d2[key]
    return dict


def callFunc(func, **kwargs):
    args = {}
    try:
        funcargs = inspect.signature(func).parameters
        for arg in funcargs.keys():
            if arg in kwargs.keys():
                args[arg] = kwargs[arg]
    except ValueError:
        pass
    return func(**args)


def evaluateArguments(args, **kwargs):
    outargs = {}
    for argkey in args.keys():
        argvalue = args[argkey]
        if(callable(argvalue)):
            outargs[argkey] = callFunc(argvalue, **kwargs)
        elif(isinstance(argvalue, list) and all(callable(item) for item in argvalue)):
            outargs[argkey] = [callFunc(item, **kwargs) for item in argvalue]
        else:
            outargs[argkey] = argvalue
    return outargs


class BatchRunner:

    def __init__(self, pipeline, outputdir=".", startCallback:Callable[[Dict], Tuple[int, widgets.Output]]|None=None, endCallback:Callable[[int, Dict, Image.Image], None]|None=None):
        self.pipeline = pipeline
        self.outputdir = outputdir
        self.argsbatch = []
        self.startCallback = startCallback
        self.endCallback = endCallback


    def appendBatchArguments(self, argdict, count=1):
        batchargs = {}
        flatargs = {}

        # Expand instances of BatchArgument into lists of items
        for arg in argdict.keys():
            if(isinstance(argdict[arg], BatchArgument)):
                batchargs[arg] = argdict[arg]()
            else:
                flatargs[arg] = argdict[arg]

        for batch in range(0, count):
            # Evaluate instances of Argument
            args = evaluateArguments(flatargs)

            # product of each combo of BatchArgument
            if(len(batchargs) > 0):
                keys, values = zip(*batchargs.items())
                for bundle in itertools.product(*values):
                    iterargs = dict(zip(keys, bundle))
                    self.argsbatch.append(mergeDict(iterargs, args))
            else:
                self.argsbatch.append(args)


    def run(self):
        stop = False
        for i, args in enumerate(self.argsbatch):
            output_index, output = self.startOutput(args)
            with output:
                try:
                    print(f"Generating {i}/{len(self.argsbatch)}")
                    self.logArgs(args)

                    image, seed = self.pipeline(**args)
                    self.argsbatch[i]["image"] = image
                    self.argsbatch[i]["seed"] = seed
                    self.argsbatch[i]["timestamp"] = int(time.time())

                    if(self.endCallback is not None):
                        self.endCallback(output_index, args, image)
                except KeyboardInterrupt:
                    stop = True
                    break
                except Exception as e:
                    print(traceback.format_exc())
                    stop = True
                    break
            if(stop):
                break



    def startOutput(self, args) -> Tuple[int, widgets.Output]:
        if (self.startCallback):
            return self.startCallback(args)
        else:
            return widgets.Output()



    def logArgs(self, args):
        print(f"Arguments: {args}")
        for arg in args.keys():
            value = args[arg]
            if (isinstance(value, Image.Image)):
                print(f"{arg}:")
                if(hasattr(value, "filename")):
                    print(getattr(value, "filename"))
                thumb = value.copy()
                thumb.thumbnail((256,256), Image.ANTIALIAS)
                display(thumb)
            elif (isinstance(value, list) and all(isinstance(item, Image.Image) for item in value)):
                print(f"{arg}:")
                for item in value:
                    if(hasattr(item, "filename")):
                        print(getattr(item, "filename"))
                    thumb = item.copy()
                    thumb.thumbnail((256,256), Image.ANTIALIAS)
                    display(thumb)
