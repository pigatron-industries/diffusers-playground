import random
import re
from .Argument import Argument
from typing import List, Dict


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
    def __init__(self, modifier_dict:Dict[str, List[str]], wildcard_dict:List[str], prompt:str="", shuffle:bool=False):
        self.modifier_dict = modifier_dict
        self.wildcard_dict = wildcard_dict
        self.prompt = prompt
        self.shuffle = shuffle

    def setPrompt(self, prompt):
        self.prompt = prompt

    def randomItemsFromDict(self, modifiername, num):
        items = self.modifier_dict[modifiername]
        return random.sample(items, num)
    
    def randomiseWildcards(self, prompt):
        out_prompt = prompt
        tokenised_brackets = re.findall(r'<.*?>', prompt)
        for prompttoken in tokenised_brackets:
            tokenname = re.sub(r'\[[^\]]*\]', '', prompttoken) # ignore everything between square brackets
            tokenregex = tokenname.replace('*', '.*')
            matchingstrings = []
            for wildcard_match in self.wildcard_dict:
                if(re.match(tokenregex, wildcard_match)):
                    matchingstrings.append(wildcard_match)
            randomstring = random.choice(matchingstrings)
            out_prompt = out_prompt.replace(prompttoken, randomstring)
        return out_prompt

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
    
    def shufflePrompt(self, prompt):
        parts = prompt.split(',')
        parts = [s.strip() for s in parts]
        random.shuffle(parts)
        return ', '.join(parts)

    def __call__(self):
        outprompt = self.prompt
        outprompt = self.randomiseFromDict(outprompt)
        outprompt = self.randomiseWildcards(outprompt)
        outprompt = self.randomiseFromPrompt(outprompt)
        if(self.shuffle):
            outprompt = self.shufflePrompt(outprompt)
        return outprompt
