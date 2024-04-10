from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.image.process import SemanticSegmentationNode
from diffuserslib.functional.nodes.user import ImageUploadInputNode
from PIL import Image


class MLSDEdgedetectionWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image - Semantic Segmentation", Image.Image, workflow=False, subworkflow=True, realtime=False)


    def build(self):
        image_input = ImageUploadInputNode(name = "image")
        segmentation = SemanticSegmentationNode(image = image_input)
        return segmentation
    