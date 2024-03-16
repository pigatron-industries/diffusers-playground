from diffuserslib.functional import *


class ImageGenerationWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Generation - Generic", Image.Image, workflow=True, subworkflow=False)


    def build(self):
        image_input = ImageSelectInputNode(name = "image_output")
        resize_image = ResizeImageNode(image = image_input, size = (512, 512), type = ResizeImageNode.ResizeType.STRETCH, name = "resize_image")
        return resize_image
