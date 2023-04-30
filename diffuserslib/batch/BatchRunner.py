import itertools
import time
from PIL import Image

from IPython.display import display
import ipywidgets as widgets
import functools

from .argument.Argument import BatchArgument


def mergeDict(d1, d2):
    dict = d1
    for key in d2.keys():
        dict[key] = d2[key]
    return dict


def evaluateArguments(args, **kwargs):
    outargs = {}
    for argkey in args.keys():
        argvalue = args[argkey]
        if(callable(argvalue)):
            outargs[argkey] = argvalue(**kwargs)
        elif(isinstance(argvalue, list) and all(callable(item) for item in argvalue)):
            outargs[argkey] = [item(**kwargs) for item in argvalue]
        else:
            outargs[argkey] = argvalue
    return outargs


class BatchRunner:

    def __init__(self, pipeline, argdict, count=1, outputdir="."):
        self.pipeline = pipeline
        self.argdict = argdict
        self.count = count
        self.outputdir = outputdir
        self._createBatchArguments()
        print(f"Created batch of size {len(self.argsbatch)}")


    def _createBatchArguments(self):
        batchargs = {}
        flatargs = {}
        self.argsbatch = []

        # Expand instances of BatchArgument into lists of items
        for arg in self.argdict.keys():
            if(isinstance(self.argdict[arg], BatchArgument)):
                batchargs[arg] = self.argdict[arg]()
            else:
                flatargs[arg] = self.argdict[arg]

        for batch in range(0, self.count):
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

        return batch


    def run(self):
        for i, args in enumerate(self.argsbatch):
            print(f"Generating {i}/{len(self.argsbatch)}")
            self.logArgs(args)

            image, seed = self.pipeline(**args)
            self.argsbatch[i]["image"] = image
            self.argsbatch[i]["seed"] = seed
            self.argsbatch[i]["timestamp"] = int(time.time())
            self._output(image, seed, i)


    def logArgs(self, args):
        print(f"Arguments: {args}")
        for arg in args.keys():
            value = args[arg]
            if (isinstance(value, Image.Image)):
                print(f"{arg}:")
                display(value)
            elif (isinstance(value, list) and all(isinstance(item, Image.Image) for item in value)):
                print(f"{arg}:")
                for item in value:
                    display(item)


    def _output(self, image, seed, index):
        self.argsbatch[index]["image"] = image
        self.argsbatch[index]["seed"] = seed
        display(image)

        output = widgets.Output()
        self.argsbatch[index]["output"] = output
        saveBtn = widgets.Button(description="Save")
        saveBtn.on_click(functools.partial(self._saveOutput, index))
        display(saveBtn, output)


    def _saveOutput(self, index, btn):
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
                elif (isinstance(value, Image.Image) and hasattr(value, 'filename')):
                    file.write(f"{arg}: {value.filename}\n")
                elif (isinstance(value, list)):
                    if (all(isinstance(item, str) for item in value) or all(isinstance(item, int) for item in value) or all(isinstance(item, float) for item in value)):
                        file.write(f"{arg}: {value}\n")