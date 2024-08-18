from diffuserslib.functional import *


class ImageZoomWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Transform - Zoom", Image.Image, workflow=False, subworkflow=True, realtime=False)


    def build(self):
        image_input = ImageUploadInputNode(name = "image")
        zoom_input = FloatUserInputNode(value = 0.9, name = "zoom")
        rotate = RotateImageNode(image = image_input, angle = zoom_input)
        return rotate
    