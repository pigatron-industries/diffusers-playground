from typing import List
import yaml
import os
from .Transforms import *
from .TransformsDiffusion import *
from .TransformsInit import *



def str_to_class(str):
    return getattr(sys.modules[__name__], str)
        

class Sequence():
    def __init__(self, name:str, length:int, transforms:List[Transform], initimage:str=None):
        self.name = name
        self.length = length
        self.initimage = initimage
        self.transforms = transforms


class Scene():
    def __init__(self, sequences: List[Sequence]):
        self.sequences = sequences

    @classmethod
    def from_file(cls, filename, pipelines):
        inputdir = os.path.dirname(filename)
        filedata = yaml.safe_load(open(filename, "r"))
        sequences = []
        for sequence in filedata['sequences']:
            transforms = []
            for transform in sequence['transforms']:
                transform['length'] = sequence['length']
                transform['pipelines'] = pipelines
                transform['inputdir'] = inputdir
                transform['transforms'] = cls._load_subtransforms(transform)
                transform['interpolation'] = cls._load_interpolation(transform)
                # TODO transfer default params to transform
                transformClass = str_to_class(f"{transform['type']}Transform")
                transforms.append(transformClass(**transform))
            sequence['transforms'] = transforms
            sequences.append(Sequence(**sequence))
        return Scene(sequences=sequences)


    @classmethod
    def _load_subtransforms(cls, transform):
        if('transforms' in transform):
            subtransforms = []
            for subtransform in transform['transforms']:
                subtransformClass = str_to_class(f"{subtransform['type']}Transform")
                subtransform['length'] = transform['length']
                subtransform['pipelines'] = transform['pipelines']
                subtransform['inputdir'] = transform['inputdir']
                subtransform['interpolation'] =cls._load_interpolation(subtransform)
                subtransforms.append(subtransformClass(**subtransform))
            return subtransforms
        else:
            return None

    @classmethod
    def _load_interpolation(cls, transform):
        if('interpolation' in transform):
            interpolationClass = str_to_class(f"{transform['interpolation']}Interpolation")
            return interpolationClass()
        else:
            return None

