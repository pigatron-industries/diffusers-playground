from typing import List
import yaml
from .Transforms import *
from .TransformsDiffusion import *



def str_to_class(str):
    return getattr(sys.modules[__name__], str)
        

class Sequence():
    def __init__(self, name:str, length:int, transforms:List[Transform]):
        self.name = name
        self.length = length
        self.transforms = transforms


class Scene():
    def __init__(self, sequences: List[Sequence]):
        self.sequences = sequences

    @classmethod
    def from_file(cls, filename, pipelines):
        filedata = yaml.safe_load(open(filename, "r"))
        sequences = []
        for sequence in filedata['sequences']:
            transforms = []
            for transform in sequence['transforms']:
                transform['length'] = sequence['length']
                transform['pipelines'] = pipelines
                if('interpolation' in transform):
                    interpolationClass = str_to_class(f"{transform['interpolation']}Interpolation")
                    transform['interpolation'] = interpolationClass()
                transformClass = str_to_class(f"{transform['type']}Transform")
                transforms.append(transformClass(**transform))
            sequence['transforms'] = transforms
            sequences.append(Sequence(**sequence))
        return Scene(sequences=sequences)
