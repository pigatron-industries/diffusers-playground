import itertools
import numpy as np
import random
import time

from IPython.display import display
import ipywidgets as widgets
import functools


class BatchArgument:
    pass


class Argument:
    pass    


class NumberRangeBatchArgument(BatchArgument):
    def __init__(self, min, max, step):
        self.min = min
        self.max = max
        self.step = step
        
    def __call__(self):
        return range(self.min, self.max, self.step)


class RandomNumberBatchArgument(BatchArgument):
    def __init__(self, min, max, num):
        self.min = min
        self.max = max
        self.num = num
        
    def __call__(self):
        return np.random.randint(self.min, self.max, self.num)


class RandomNumberArgument(Argument):
    def __init__(self, min, max):
        self.min = min
        self.max = max
        
    def __call__(self):
        return random.randint(self.min, self.max)


class StringListBatchArgument(BatchArgument):
    def __init__(self, list):
        self.list = list
    
    def __call__(self):
        return self.list



class DiffusersBatch:

    def __init__(self, pipeline, argdict, outputdir):
        self.pipeline = pipeline
        self.argdict = argdict
        self.outputdir = outputdir
        self._createBatchArguments()
        print(f"Created batch of size {len(self.argsbatch)}")


    def _createBatchArguments(self):
        batchargs = {}
        flatargs = {}
        batch = []
        for arg in self.argdict.keys():
            if(isinstance(self.argdict[arg], BatchArgument)):
                batchargs[arg] = self.argdict[arg]()
            else:
                flatargs[arg] = self.argdict[arg]
        
        keys, values = zip(*batchargs.items())
        for bundle in itertools.product(*values):
            args = dict(zip(keys, bundle))
            for flatargkey in flatargs.keys():
                if(isinstance(flatargs[flatargkey], Argument)):
                    args[flatargkey] = flatargs[flatargkey]()
                else:
                    args[flatargkey] = flatargs[flatargkey]
            batch.append(args)

        self.argsbatch = batch
        return batch


    def run(self):
        for i, args in enumerate(self.argsbatch):
            print(f"Running for arguments: {args}")
            image, seed = self.pipeline(**args)
            self.argsbatch[i]["image"] = image
            self.argsbatch[i]["seed"] = seed
            self.argsbatch[i]["timestamp"] = int(time.time())
            self._output(image, seed, i)


    def _output(self, image, seed, index):
        self.argsbatch[index]["image"] = image
        self.argsbatch[index]["seed"] = seed
        display(image)

        output = widgets.Output()
        self.argsbatch[index]["output"] = output
        saveBtn = widgets.Button(description="Save")
        saveBtn.on_click(functools.partial(self._saveImage, index))
        display(saveBtn, output)


    def _saveImage(self, index, btn):
        args = self.argsbatch[index]
        timestamp = args['timestamp']
        output = args['output']
        image = args['image']
        image_filename = f"{self.outputdir}/txt2img_{timestamp}.png"
        info_filename = f"{self.outputdir}/txt2img_{timestamp}.txt"
        image.save(image_filename)
        self._saveArgs(args, info_filename)
        output.append_stdout("Saved to: " + image_filename)

    def _saveArgs(self, args, file):
        with open(file, 'w') as file:
            for arg in args.keys():
                value = args[arg]
                if (isinstance(value, str) or isinstance(value, int) or isinstance(value, float)):
                    file.write(f"{arg}: {value}\n")