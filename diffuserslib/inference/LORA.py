from .arch.StableDiffusionPipelines import DiffusersPipelineWrapper
from typing import Dict, List, Tuple
import re
import torch
import json
import os
from safetensors import safe_open



class LORA:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.triggers: List[str] = []

    def load_triggers(self, top_n: int = 5):
        try:
            self.triggers = guess_trigger_words(self.path, top_n)
        except Exception:
            self.triggers = []

    @classmethod
    def from_file(cls, name, path):
        lora = cls(name, path)
        # lora.load_triggers()
        # print(f"Loaded LORA {name} from {path} with triggers: {lora.triggers}")
        return lora


def read_safetensors_metadata(path: str) -> dict:
    # Return metadata dict for a safetensors file, or empty dict if unavailable
    if not path or not os.path.exists(path):
        return {}
    if path.endswith('.safetensors'):
        try:
            with safe_open(path, framework='pt') as f:
                # safetensors SafeFile exposes metadata as .metadata
                meta = getattr(f, 'metadata', None)
                if meta is None:
                    return {}
                # metadata values may be bytes; convert to str
                out = {}
                for k, v in meta.items():
                    if isinstance(v, (bytes, bytearray)):
                        try:
                            out[k] = v.decode('utf-8')
                        except Exception:
                            out[k] = str(v)
                    else:
                        out[k] = v
                return out
        except Exception:
            return {}
    else:
        return {}


def guess_trigger_words(path: str, top_n: int = 5) -> list[str]:
    meta = read_safetensors_metadata(path)
    # Civitai's explicit field, if present
    if "modelspec.trigger_phrase" in meta:
        return [meta["modelspec.trigger_phrase"]]

    # Kohya tag frequency — most common tags as candidates
    if "ss_tag_frequency" in meta:
        try:
            tag_freq = json.loads(meta["ss_tag_frequency"])
        except Exception:
            return []
        counts: dict[str, int] = {}
        for dataset in tag_freq.values():  # keyed by dataset dir
            for tag, count in dataset.items():
                counts[tag] = counts.get(tag, 0) + count
        return [t for t, _ in sorted(counts.items(), key=lambda x: -x[1])[:top_n]]

    return []
        

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

