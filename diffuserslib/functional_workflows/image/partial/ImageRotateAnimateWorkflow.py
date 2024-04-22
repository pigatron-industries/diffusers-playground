from diffuserslib.functional import *


class ImageRotateAnimateWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Transform - Rotation Animate", Image.Image, workflow=False, subworkflow=True, realtime=False)


    def build(self):
        time_delta_user_input = FloatUserInputNode(value = 0.01, name = "time_delta")
        image_input = ImageUploadInputNode(name = "image")

        angle_animate = AnimateFloatNode(dt=time_delta_user_input)
        
        rotate = RotateImageNode(image = image_input, angle = angle_animate)
        return rotate