from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.util.ModuleLoader import ModuleLoader




class ImageToPezNode(FunctionalNode):
    def __init__(self, 
                 image:ImageFuncType,
                 name:str = "image_to_pez"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        

    def process(self, image:Image.Image) -> str:
        ModuleLoader.load_from_directory("workspace/hard-prompts-made-easy/open_clip", recursive=True)
        return "Not implemented yet."
