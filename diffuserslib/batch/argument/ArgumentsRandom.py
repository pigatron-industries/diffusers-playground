import random
import glob
from PIL import Image
from .Argument import Argument


class RandomNumberArgument(Argument):
    """ Select a random number between min and max """
    def __init__(self, min, max):
        self.min = min
        self.max = max
        
    def __call__(self, **kwargs):
        return random.randint(self.min, self.max)


class SequentialNumberArgument(Argument):
    """  """
    def __init__(self, start):
        self.num = start
        
    def __call__(self, **kwargs):
        num = self.num
        self.num = self.num + 1
        return num


class RandomChoiceArgument(Argument):
    def __init__(self, list):
        self.list = list

    def __call__(self, **kwargs):
        return random.choice(self.list)


class RandomPositionArgument(Argument):
    """ Get a random position in an image """
    def __init__(self, border_width = 32):
        self.border_width = border_width

    def __call__(self, context):
        left = self.border_width
        top = self.border_width
        right = context.size[0] - self.border_width
        bottom = context.size[1] - self.border_width
        return (random.randint(left, right), random.randint(top, bottom))


class RandomImage(Argument):

    @classmethod
    def fromDirectory(cls, directory):
        filelist = glob.glob(f'{directory}/*.png')
        return cls(filelist)

    def __init__(self, filelist):
        self.filelist = filelist

    def __call__(self, **kwargs):
        file = random.choice(self.filelist)
        image = Image.open(file)
        image.filename = file
        return image


class RandomImageSelection(Argument):

    @classmethod
    def fromDirectory(cls, directory):
        filelist = glob.glob(f'{directory}/*.png')
        return cls(filelist)

    def __init__(self, filelist, width, height, rotate, crop, **kwargs):
        self.filelist = filelist

    def __call__(self):
        file = random.choice(self.filelist)
        image = Image.open(file)
        image.filename = file
        return image