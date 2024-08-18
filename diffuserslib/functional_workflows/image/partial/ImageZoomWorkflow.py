from diffuserslib.functional import *
from diffuserslib.functional.nodes.image.transform import ZoomImageNode


class ImageZoomWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Transform - Zoom", Image.Image, workflow=False, subworkflow=True, realtime=False)


    def build(self):
        image_input = ImageUploadInputNode(name = "image")
        zoom_input = FloatUserInputNode(value = 1.05, name = "zoom")
        rotate = ZoomImageNode(image = image_input, zoom = zoom_input)
        return rotate
    