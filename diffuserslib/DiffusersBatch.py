

import itertools
import numpy as np
import random


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


class RandomNumberArgument(Argument):
    def __init__(self, min, max):
        self.min = min
        self.max = max
        
    def __call__(self):
        return random.randint(self.min, self.max)


class StringListBatchArgument(BatchArgument):
    def __init__(self, list):
        self.list = list
    
    def __call__(self):
        return self.list


def createBatchArguments(argdict):
    batchargs = {}
    flatargs = {}
    batch = []
    for arg in argdict.keys():
        if(isinstance(argdict[arg], BatchArgument)):
            batchargs[arg] = argdict[arg]()
        else:
            flatargs[arg] = argdict[arg]
    
    keys, values = zip(*batchargs.items())
    for bundle in itertools.product(*values):
        args = dict(zip(keys, bundle))
        for flatargkey in flatargs.keys():
            if(isinstance(flatargs[flatargkey], Argument)):
                args[flatargkey] = flatargs[flatargkey]()
            else:
                args[flatargkey] = flatargs[flatargkey]
        batch.append(args)

    return batch

