from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from PIL import Image
import random
import os


class RandomImageNode(FunctionalNode):
    def __init__(self, paths:ListFuncType, name:str = "random_image"):
        super().__init__(name)
        self.addParam("paths", paths, List[str])


    def process(self, paths:List[str]) -> Image.Image:
        print(paths)
        random_path = random.choice(paths)
        if(os.path.isdir(random_path)):
            random_path = self.getRandomChildPath(random_path)
        return Image.open(random_path)
            

    def getRandomChildPath(self, path:str) -> str:
        childpaths = os.listdir(path)
        randompath = random.choice(childpaths)
        fullpath = os.path.join(path, randompath)
        if(os.path.isdir(fullpath)):
            return self.getRandomChildPath(fullpath)
        else:
            return fullpath
        