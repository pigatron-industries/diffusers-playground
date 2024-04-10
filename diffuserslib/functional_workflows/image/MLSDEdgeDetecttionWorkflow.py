from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.image.process import MLSDStraightLineDetectionNode
from diffuserslib.functional.nodes.user import ImageUploadInputNode
from PIL import Image


class MLSDEdgedetectionWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image - MLSD Edge Detection", Image.Image, workflow=False, subworkflow=True, realtime=False)


    def build(self):
        image_input = ImageUploadInputNode(name = "image")
        mlsd = MLSDStraightLineDetectionNode(image = image_input)
        return mlsd
    