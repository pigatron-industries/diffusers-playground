from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import GenerationParameters
from diffuserslib.inference.DiffusersUtils import tiledProcessorCentred, tiledGeneration
from .ImageDiffusionNode import ModelsFuncType, LorasFuncType, ModelsType, LorasType
from .ConditioningInputNode import ConditioningInputType, ConditioningInputFuncsType
from PIL import Image


class TileSizeCalculatorNode(FunctionalNode):
    """ 
    Calculates the size of a tile based on the total image length, max tile length, and overlap. 
    The size is calculated such that a whole number of tiles fits into the image.
    """

    def __init__(self,
                 image:ImageFuncType|None = None,
                 image_size:SizeType|None = None,
                 overlap:IntFuncType = 128,
                 max:int = 1152,
                 name:str = "tile_size"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("image_size", image_size, Tuple[int, int])
        self.addParam("overlap", overlap, int)
        self.addParam("max", max, int)


    def process(self, image:Image.Image|None, image_size:Tuple[int, int]|None, overlap:int, max:int) -> SizeType:
        width = image_size[0] if image_size is not None else image.width if image is not None else 0
        height = image_size[1] if image_size is not None else image.height if image is not None else 0
        tilewidth = TileSizeCalculatorNode.calcTileSize(width, max, overlap)
        tileheight = TileSizeCalculatorNode.calcTileSize(height, max, overlap)
        return (tilewidth, tileheight)
        

    @staticmethod
    def calcTileSize(total_length:int, max_tile_length:int, tile_overlap:int, restrict:str|None = None) -> int:
        if restrict is not None and restrict == "even":
            n = 2
        else:
            n = 1
        while True:
            tile_length = (total_length + (n - 1) * tile_overlap) // n
            if tile_length <= max_tile_length:
                tile_length = (tile_length + 7) // 8 * 8  # Round up to the nearest multiple of 8
                return tile_length
            n += 2 if restrict is not None else 1