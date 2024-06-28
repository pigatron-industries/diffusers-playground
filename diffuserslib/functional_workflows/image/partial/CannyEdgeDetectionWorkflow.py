from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.image.process import CannyEdgeDetectionNode
from diffuserslib.functional.nodes.user import ImageUploadInputNode
from PIL import Image


class CannyEdgeDetectionWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Process - Canny Edge Detection", Image.Image, workflow=False, subworkflow=True, realtime=False)

    def build(self):
        image_input = ImageUploadInputNode(name = "image")
        mlsd = CannyEdgeDetectionNode(image = image_input)
        return mlsd
    