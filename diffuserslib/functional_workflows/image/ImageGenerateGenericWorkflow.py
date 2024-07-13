from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.image.process import *
from diffuserslib.functional.nodes.image.transform.NoOpImageNode import *


class ImageGenerationGenericWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Generation - Generic", Image.Image, workflow=True, subworkflow=False)


    def build(self):
        image_input = ImageUploadInputNode(name = "image_output")
        placeholder_image = NoOpImageNode(image = image_input, name = "placeholder")
        return placeholder_image
