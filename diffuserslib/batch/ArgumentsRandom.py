import random
import re
import glob
from PIL import Image

from .BatchRunner import Argument


class RandomNumberArgument(Argument):
    """ Select a random number between min and max """
    def __init__(self, min, max):
        self.min = min
        self.max = max
        
    def __call__(self):
        return random.randint(self.min, self.max)


class RandomPromptProcessor(Argument):
    """ Randomise parts of a text prompt """
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
        image = Image.open(file)
        image.filename = file
        return image


class RandomImageSelection(Argument):

    @classmethod
    def fromDirectory(cls, directory):
        filelist = glob.glob(f'{directory}/*.png')
        return cls(filelist)

    def __init__(self, filelist, width, height, rotate, crop):
        self.filelist = filelist

    def __call__(self):
        file = random.choice(self.filelist)
        image = Image.open(file)
        image.filename = file
        return image