from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.image.process import SemanticSegmentationNode
from diffuserslib.functional.nodes.user import ImageUploadInputNode, TextAreaLinesInputNode
from PIL import Image


class SemanticSegmentationMaskWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Process - Semantic Segmentation Mask", Image.Image, workflow=False, subworkflow=True, realtime=False)


    def build(self):
        image_input = ImageUploadInputNode(name = "image")
        mask_labels_input = TextAreaLinesInputNode(name = "mask_labels")
        mask = SemanticSegmentationNode(image = image_input, mask_labels = mask_labels_input)
        return mask
    