from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.GlobalConfig import GlobalConfig
from transformers import SamModel, SamProcessor
import torch
import numpy as np


class SegmentAnythingNode(FunctionalNode):

    def __init__(self, 
                 image:ImageFuncType, 
                 name:str="segment_anything"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.model:SamModel|None = None
        self.processor:SamProcessor|None = None


    def loadModel(self) -> Tuple[SamModel, SamProcessor]:
        if(self.model is None or self.processor is None):
            self.model = SamModel.from_pretrained("facebook/sam-vit-huge").to(GlobalConfig.device) 
            self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge") 
        return self.model, self.processor
        

    def process(self, image:Image.Image) -> Image.Image:
        model, processor = self.loadModel()
        image = image.convert("RGB")
        input_points = []
        inputs = processor(image, input_points=input_points, return_tensors="pt").to(GlobalConfig.device)
        with torch.no_grad():
            outputs = model(**inputs)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )

        return masks[0]
