import numpy as np

from .BatchRunner import BatchArgument



class NumberRangeBatchArgument(BatchArgument):
    """ Iterate over a range of numbers """
    def __init__(self, min, max, step):
        self.min = min
        self.max = max
        self.step = step
        
    def __call__(self):
        return range(self.min, self.max, self.step)


class RandomNumberBatchArgument(BatchArgument):
    """ Returns a list of random numbers to iterate over """
    def __init__(self, min, max, num):
        self.min = min
        self.max = max
        self.num = num
        
    def __call__(self):
        return np.random.randint(self.min, self.max, self.num)


""" Iterate over a list of strings """
class StringListBatchArgument(BatchArgument):
    def __init__(self, list):
        self.list = list
    
    def __call__(self):
        return self.list