import itertools
import numpy as np
import random
import time
import re
import glob
from PIL import Image

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


class StringListBatchArgument(BatchArgument):
    def __init__(self, list):
        self.list = list
    
    def __call__(self):
        return self.list


class RandomNumberArgument(Argument):
    def __init__(self, min, max):
        self.min = min
        self.max = max
        
    def __call__(self):
        return random.randint(self.min, self.max)



class RandomPromptProcessor(Argument):
    def __init__(self, modifier_dict, prompt=""):
        self.modifier_dict = modifier_dict
        self.prompt = prompt

    def setPrompt(self, prompt):
        self.prompt = prompt

    def randomiseFromDict(self, prompt):
        # randomise from dictionary of items defined outside of prompt _colour_
        out_prompt = prompt

        tokenised_brackets = re.findall(r'_.*?_', out_prompt)
        for bracket in tokenised_brackets:
            modifiername = bracket[1:-1]
            options = self.modifier_dict[modifiername]
            randomoption = random.choice(options)
            out_prompt = out_prompt.replace(bracket, randomoption, 1)

        tokenised_brackets = re.findall(r'__.*?__', out_prompt)
        for bracket in tokenised_brackets:
            modifiername = bracket[1:-1]
            options = self.modifier_dict[modifiername]
            out_prompt = out_prompt.replace(f'__{modifiername}__', self.randomCombo(options))

        return out_prompt

    def randomCombo(wordlist):
        numberofwords = random.randint(0, len(wordlist)-1)
        out_prompt = ""
        random.shuffle(wordlist)
        for i, word in enumerate(wordlist):
            if i > 0:
                out_prompt += ", "
            out_prompt += word
            if i >= numberofwords:
                break
        return out_prompt

    def randomiseFromPrompt(self, prompt):
        # randomise from list of items in prompt between brackets {cat|dog}
        out_prompt = prompt
        tokenised_brackets = re.findall(r'\{.*?\}', out_prompt)
        for bracket in tokenised_brackets:
            options = bracket[1:-1].split('|')
            randomoption = random.choice(options)
            out_prompt = out_prompt.replace(bracket, randomoption, 1)
        return out_prompt

    def __call__(self):
        outprompt = self.randomiseFromDict(self.prompt)
        outprompt = self.randomiseFromPrompt(outprompt)
        return outprompt


class RandomImage(Argument):

    @classmethod
    def fromDirectory(cls, directory):
        filelist = glob.glob(f'{directory}/*.png')
        return cls(filelist)

    def __init__(self, filelist):
        self.filelist = filelist

    def __call__(self):
        file = random.choice(self.filelist)
        return Image.open(file)


def mergeDict(d1, d2):
    dict = d1
    for key in d2.keys():
        dict[key] = d2[key]
    return dict


class DiffusersBatch:

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
            args = {}
            for flatargkey in flatargs.keys():
                if(isinstance(flatargs[flatargkey], Argument)):
                    args[flatargkey] = flatargs[flatargkey]()
                else:
                    args[flatargkey] = flatargs[flatargkey]

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
            print(f"Arguments: {args}")
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