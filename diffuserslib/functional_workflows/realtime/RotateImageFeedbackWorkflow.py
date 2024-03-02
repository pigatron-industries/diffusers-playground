from diffuserslib.functional import *


class RotateImageFeedbackWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Rotation - Feedback", Image.Image, workflow=True, subworkflow=False, realtime=True)


    def build(self):
        image_input = ImageSelectInputNode(name = "image")
        angle_delta_user_input = FloatUserInputNode(value = 3.6, name = "angle_delta")

        feedback_image = FeedbackNode(init_value = image_input)
        rotate = RotateImageNode(image = feedback_image, angle = angle_delta_user_input)
        crop = ResizeImageNode(image = rotate, size = (512, 512), type = ResizeImageNode.ResizeType.EXTEND)
        feedback_image.setInput(crop)
        return crop
    