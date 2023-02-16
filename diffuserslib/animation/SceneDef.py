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
    def __init__(self, name:str, length:int, transforms:List[Transform], init:List[Transform]=None, inputdir:str=None):
        self.name = name
        self.length = length
        self.init = init
        self.transforms = transforms

    @classmethod
    def from_params(cls, params, defaults):
        # initial frame transform
        inittransforms = []
        if('init' in params):
            for transform in params['init']:
                transform['length'] = 1
                addDefaultParams(transform, defaults)
                inittransforms.append(loadObject(transform, "Transform"))
            params['init'] = inittransforms
        # animation transforms
        transforms = []
        for transform in params['transforms']:
            addDefaultParams(transform, defaults)
            transform['length'] = params['length']
            transform['transforms'] = cls._load_subtransforms(transform)
            transform['interpolation'] = cls._load_interpolation(transform)
            transforms.append(loadObject(transform, "Transform"))
        params['transforms'] = transforms

        params['inputdir'] = defaults['inputdir']
        return cls(**params)

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



class Scene():
    def __init__(self, sequences:List[Sequence], inputdir:str):
        self.sequences = sequences
        self.inputdir = inputdir

    @classmethod
    def from_file(cls, filename, pipelines):
        inputdir = os.path.dirname(filename)
        filedata = yaml.safe_load(open(filename, "r"))
        defaults = filedata['defaults']
        defaults['inputdir'] = inputdir
        defaults['pipelines'] = pipelines
        sequences = []
        for sequence in filedata['sequences']:
            sequences.append(Sequence.from_params(sequence, defaults))
        return Scene(sequences=sequences, inputdir=inputdir)



def loadObject(params, classPostfix):
    classname = params['type'] + classPostfix
    cls = getattr(sys.modules[__name__], classname)
    return cls(**params)
