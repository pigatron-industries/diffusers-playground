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
        image = Image.open(random_path)
        setattr(image, "file", random_path)
        return image
            

    def getRandomChildPath(self, path:str) -> str:
        childpaths = []
        with os.scandir(path) as entries:
            for entry in entries:
                if(entry.is_dir()):
                    childpaths.append(entry.name)
                elif entry.is_file() and entry.name.endswith((".png", ".jpg", ".jpeg")):
                    childpaths.append(entry.name)
        randompath = random.choice(childpaths)
        fullpath = os.path.join(path, randompath)
        if(os.path.isdir(fullpath)):
            return self.getRandomChildPath(fullpath)
        else:
            return fullpath
        