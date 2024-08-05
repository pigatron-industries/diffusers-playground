from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.image.process import ImageCompositeNode
from diffuserslib.functional.nodes.user import ImageUploadInputNode, TextAreaLinesInputNode
from PIL import Image


class ImageCompositeWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Process - Composite", Image.Image, workflow=False, subworkflow=True, realtime=False)


    def build(self):
        foreground_input = ImageUploadInputNode(display = "Foreground", name = "foreground")
        background_input = ImageUploadInputNode(display = "Background", name = "background")
        mask_input = ImageUploadInputNode(display = "Mask", name = "mask")
        composite = ImageCompositeNode(foreground = foreground_input, background = background_input, mask = mask_input)
        return composite
    