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
        
    def __call__(self, **kwargs):
        return random.randint(self.min, self.max)


class SequentialNumberArgument(Argument):
    """  """
    def __init__(self, start):
        self.num = start
        
    def __call__(self, **kwargs):
        num = self.num
        self.num = self.num + 1
        return num


class RandomChoiceArgument(Argument):
    def __init__(self, list):
        self.list = list

    def __call__(self, **kwargs):
        return random.choice(self.list)


class RandomPositionArgument(Argument):
    """ Get a random position in an image """
    def __init__(self, border_width = 32):
        self.border_width = border_width

    def __call__(self, context):
        left = self.border_width
        top = self.border_width
        right = context.size[0] - self.border_width
        bottom = context.size[1] - self.border_width
        return (random.randint(left, right), random.randint(top, bottom))


class RandomPromptProcessor(Argument):
    """ Randomise parts of a text prompt 
            randomise from list of items in prompt between brackets 
                {cat|dog}
            randomise from dictionary of items defined outside of prompt, using single undercore format: 
                _colour_  =  "red"
            pick a number of items from dictionary:
                _colour[3]_  =  "red green blue"
            pick a random number of itesm between two numbers: 
                _colour[2-3]_ 
            change delimiter for chosen items: 
                _colour[2-3,]_  =  "red, green, blue"
                _colour[2-3 and]_  =  "red and green and blue"


    """
    def __init__(self, modifier_dict, prompt=""):
        self.modifier_dict = modifier_dict
        self.prompt = prompt

    def setPrompt(self, prompt):
        self.prompt = prompt

    def randomItemsFromDict(self, modifiername, num):
        items = self.modifier_dict[modifiername]
        return random.sample(items, num)

    def randomiseFromDict(self, prompt):
        # randomise from dictionary of items defined outside of prompt _colour_
        out_prompt = prompt
        tokenised_brackets = re.findall(r'_.*?_', out_prompt)
        for token in tokenised_brackets:
            modifiername = re.sub(r'\[[^\]]*\]|_', '', token) #remove underscores and everything between square brackets
            options = re.findall(r'\[(.*?)\]', token) # get everything between square brackets
            delimiter = ' '
            if(len(options) > 0):
                numbers = [int(x) for x in re.findall(r'\d+', options[0])]
                if(len(numbers) == 1): 
                    num = numbers[0]
                else:
                    num = random.randint(numbers[0], numbers[1])
                delimiterMatch = re.search(r'\d+([^\d]+)$', options[0]) # get all non numerical chars after last numerical
                if(delimiterMatch):
                    delimiter = delimiterMatch.group(1) + ' '
            else:
                num = 1
            items = self.randomItemsFromDict(modifiername, num)
            out_prompt = out_prompt.replace(token, delimiter.join(items), 1)
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

    def __call__(self, **kwargs):
        file = random.choice(self.filelist)
        image = Image.open(file)
        image.filename = file
        return image


class RandomImageSelection(Argument):

    @classmethod
    def fromDirectory(cls, directory):
        filelist = glob.glob(f'{directory}/*.png')
        return cls(filelist)

    def __init__(self, filelist, width, height, rotate, crop, **kwargs):
        self.filelist = filelist

    def __call__(self):
        file = random.choice(self.filelist)
        image = Image.open(file)
        image.filename = file
        return image