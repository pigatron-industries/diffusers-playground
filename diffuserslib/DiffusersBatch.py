

import itertools


class BatchArgument:
    pass
        

class NumberRangeBatchArgument(BatchArgument):
    def __init__(self, min, max, step):
        self.min = min
        self.max = max
        self.step = step
        
    def __call__(self):
        return range(self.min, self.max, self.step)



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
        args = args | flatargs
        batch.append(args)

    return batch

