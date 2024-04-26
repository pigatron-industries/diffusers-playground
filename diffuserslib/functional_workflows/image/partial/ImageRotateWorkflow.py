from diffuserslib.functional import *


class ImageRotateWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Transform - Rotation", Image.Image, workflow=False, subworkflow=True, realtime=False)


    def build(self):
        image_input = ImageUploadInputNode(name = "image")
        angle_input = FloatUserInputNode(value = 3.6, name = "angle")
        rotate = RotateImageNode(image = image_input, angle = angle_input)
        return rotate
    