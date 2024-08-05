from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib import pilToCv2
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from enum import Enum
import torch
import numpy as np



class SemanticSegmentationNode(FunctionalNode):

    OUTPUT_TYPES = ["palette", "mask"]

    def __init__(self, 
                 image:ImageFuncType, 
                 mask_labels:StringsFuncType|None = None,
                 name:str="semantic_segmentation"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("mask_labels", mask_labels, List[str])
        self.model = None
        

    def process(self, image:Image.Image, mask_labels:List[str]|None) -> Image.Image:
        seg = self.inference(image)

        palette = np.array(self.ade_palette())
        output_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

        for label, color in enumerate(palette):
            if (np.any(np.array(seg) == label)):
                text = self.ade_palette_names()[label]
                print(f"Label: {text}, Color: {color}")
            if(mask_labels is None):
                output_seg[seg == label, :] = color
            elif(self.ade_palette_names()[label] in mask_labels):
                output_seg[seg == label, :] = [255, 255, 255]


        return Image.fromarray(output_seg)
            

    def inference(self, image:Image.Image):
        if self.model is None:
            self.model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-large") # type: ignore
        image = image.convert("RGB")
        processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-swin-large")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.model(pixel_values)
        seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        return seg


    @staticmethod
    def ade_palette_names():
        """ADE20K palette class names."""
        return [
            "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "window", "grass",
            "cabinet", "pavement", "person", "ground", "door", "table", "mountain", "plant", "curtain",
            "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "carpet",
            "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing",
            "cushion", "base", "box", "column", "sign", "drawers", "counter", "sand", "sink", "skyscraper",
            "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pooltable",
            "pillow", "screendoor", "stairway", "river", "bridge", "bookcase", "blind", "coffeetable",
            "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchenisland",
            "computer", "swivelchair", "boat", "bar", "arcade", "hut", "bus", "towel", "light", "truck",
            "tower", "chandelier", "awning", "streetlight", "booth", "television", "airplane", "dirt",
            "clothes", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster",
            "stage", "van", "ship", "fountain", "conveyorbelt", "canopy", "washer", "toy", "swimmingpool",
            "stool", "barrel", "basket", "waterfall", "tent", "bag", "motorbike", "cradle", "oven", "ball",
            "food", "step", "tank", "trade", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
            "screen", "blanket", "sculpture", "hood", "sconce", "vase", "trafficlight", "tray", "bin", "fan",
            "pier", "screen", "plate", "monitor", "bulletinboard", "shower", "radiator", "glass", "clock", "flag"
        ]


    @staticmethod
    def ade_palette():
        """ADE20K palette that maps each class to RGB values."""
        return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]