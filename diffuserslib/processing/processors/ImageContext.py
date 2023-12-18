from typing import Tuple, Optional
from PIL import Image
from IPython.display import display


class ImageContext():
    def __init__(self, size:Tuple[int, int]|None, oversize:int=256):
        self.oversize = oversize
        self.size = size
        self.offset = (oversize, oversize)
        self.filename = None
        self.image = []
        if(size is not None):
            self.viewport = (oversize, oversize, oversize+size[0], oversize+size[1])
            self.image.append(Image.new("RGBA", size=(size[0]+oversize*2, size[1]+oversize*2)))
        else:
            self.viewport = (oversize, oversize, oversize, oversize)


    def getFullImage(self) -> Image.Image:
        if(len(self.image) > 0):
            return self.image[-1].copy()
        else:
            raise Exception("No image in context")
    

    def setFullImage(self, image:Image.Image):
        self.image.append(image.copy())
        self.calcSize()


    def getViewportImage(self) -> Image.Image:
        if(len(self.image) > 0):
            image = self.image[-1].crop(self.viewport)
            if(self.filename is not None):
                setattr(image, "filename", self.filename)
            return image
        else:
            raise Exception("No image in context")


    def setViewportImage(self, image:Image.Image):
        newImage = Image.new("RGBA", size=(image.width+self.oversize*2, image.height+self.oversize*2))
        newImage.paste(image, self.offset)
        self.image.append(newImage)
        if(hasattr(image, "filename")):
            self.filename = getattr(image, "filename")
        self.calcSize()

    def calcSize(self):
        if(len(self.image) > 0):
            self.size = (self.image[-1].width-self.oversize*2, self.image[-1].height-self.oversize*2)
            self.viewport = (self.oversize, self.oversize, self.oversize+self.size[0], self.oversize+self.size[1])