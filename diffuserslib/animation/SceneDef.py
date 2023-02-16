from typing import List
import yaml
import os
from PIL import Image
from .Transforms import *
from .TransformsDiffusion import *
from .TransformsInit import *


def addDefaultParams(dict, defaults):
    for key in defaults:
        if (key not in dict):
            dict[key] = defaults[key]
    return dict
        

class Sequence():
    def __init__(self, name:str, length:int, transforms:List[Transform], initimage:str=None, inputdir:str=None):
        self.name = name
        self.length = length
        if(initimage is not None):
            self.initimage = Image.open(f"{inputdir}/{initimage}")
        else:
            self.initimage = None
        self.transforms = transforms


class Scene():
    def __init__(self, sequences:List[Sequence], inputdir:str):
        self.sequences = sequences
        self.inputdir = inputdir


    @classmethod
    def from_file(cls, filename, pipelines):
        inputdir = os.path.dirname(filename)
        filedata = yaml.safe_load(open(filename, "r"))
        defaults = filedata['defaults']
        sequences = []
        for sequence in filedata['sequences']:
            transforms = []
            for transform in sequence['transforms']:
                transform['length'] = sequence['length']
                transform['pipelines'] = pipelines
                transform['inputdir'] = inputdir
                transform['transforms'] = cls._load_subtransforms(transform)
                transform['interpolation'] = cls._load_interpolation(transform)
                addDefaultParams(transform, defaults)
                transforms.append(loadObject(transform, "Transform"))
            sequence['transforms'] = transforms
            sequence['inputdir'] = inputdir
            sequences.append(Sequence(**sequence))
        return Scene(sequences=sequences, inputdir=inputdir)


    @classmethod
    def _load_subtransforms(cls, transform):
        if('transforms' in transform):
            subtransforms = []
            for subtransform in transform['transforms']:
                subtransform['length'] = transform['length']
                subtransform['pipelines'] = transform['pipelines']
                subtransform['inputdir'] = transform['inputdir']
                subtransform['interpolation'] =cls._load_interpolation(subtransform)
                subtransforms.append(loadObject(subtransform, "Transform"))
            return subtransforms
        else:
            return None

    @classmethod
    def _load_interpolation(cls, transform):
        if('interpolation' in transform):
            interpolation = {}
            interpolation['type'] = transform['interpolation']
            return loadObject(interpolation, "Interpolation")
        else:
            return None


def loadObject(params, classPostfix):
    classname = params['type'] + classPostfix
    cls = getattr(sys.modules[__name__], classname)
    return cls(**params)
