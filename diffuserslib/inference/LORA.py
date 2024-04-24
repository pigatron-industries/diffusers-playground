from .arch.StableDiffusionPipelines import DiffusersPipelineWrapper
from typing import Dict, List, Tuple
import re
import torch



class LORA:
    def __init__(self, name, path):
        self.name = name
        self.path = path

    @classmethod
    def from_file(cls, name, path):
        return cls(name, path)
        

class LORAs:
    def __init__(self):
        self.loras: Dict[str, LORA] = {}

    def __getitem__(self, name: str) -> LORA:
        return self.loras[name]
    
    def __setitem__(self, name: str, lora: LORA):
        self.loras[name] = lora

    def add(self, lora: LORA):
        self.loras[lora.name] = lora

    def keys(self):
        return self.loras.keys()
    
    def process_prompt_and_add_loras(self, prompt:str, pipeline: DiffusersPipelineWrapper, loras:List[LORA], weights:List[float]):
        prompt, prompt_loras, prompt_weights = self.process_prompt(prompt)
        loras.extend(prompt_loras)
        weights.extend(prompt_weights)
        pipeline.add_loras(loras, weights)
        return prompt
    

    def get_lorastrings_from_prompt(self, prompt:str):
        return re.findall(r'<lora:.*?>', prompt)
    

    def process_prompt(self, prompt: str) -> Tuple[str, List[LORA], List[float]]:
        lorastrings = self.get_lorastrings_from_prompt(prompt)
        loras = []
        weights = []
        for lorastring in lorastrings:
            lorastringparts = lorastring[1:-1].split(':')  # remove < and > and split by :
            loraname = lorastringparts[1]
            weight = float(lorastringparts[2]) if len(lorastringparts) > 2 else 1.0
            if('*' in loraname):
                lora = self.randomize_wildcard_lora(loraname)
            else:
                lora = self.loras[loraname]
            loras.append(lora)
            weights.append(weight)
            prompt = prompt.replace(lorastring, '')  # remove lora string from prompt
        return prompt, loras, weights
    

    def randomize_wildcard_lora(self, loraname: str) -> LORA:
        """ Replace wildcard * with random lora """
        loraregex = loraname.replace('*', '.*')
        print(f"Randomizing wildcard lora {loraname} with regex {loraregex}")
        matchingloras = []
        for lora in self.loras.values():
            if re.match(loraregex, lora.name):
                matchingloras.append(lora)
        if(len(matchingloras) > 0):
            return matchingloras[torch.randint(len(matchingloras), (1,))]
        else:
            raise ValueError(f"Could not find any lora token matching wildcard {loraname}")

