from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.types import Video
from diffuserslib.functional.nodes.animated import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.video.diffusers import *
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.image.transform import *


class VideoDiffusionWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video Diffusion - Generate", Video, workflow=True, subworkflow=True)

    def build(self):
        model_input = DiffusionModelUserInputNode(basemodels=["svd_1_0", "cogvideox_t2v"], modeltype="video", name="model")
        size_input = SizeUserInputNode(value = (512, 512), name="size")
        seed_input = SeedUserInputNode(value = None, name="seed")
        frames_input = IntUserInputNode(value = 16, name = "frames")
        fps_input = IntUserInputNode(value = 7, name = "fps")

        image_input = ImageUploadInputNode(name = "image")

        resize_image = ResizeImageNode(image = image_input, size = size_input, type = ResizeImageNode.ResizeType.STRETCH)
        conditioning_inputs = [ConditioningInputNode(image = resize_image, model = "initimage")]

        video_diffusion = VideoDiffusionStableVideoDiffusionNode(models = model_input, 
                                                            size = size_input, 
                                                            seed = seed_input,
                                                            frames = frames_input,
                                                            conditioning_inputs = conditioning_inputs)
        frames_to_video = FramesToVideoNode(frames = video_diffusion, fps = fps_input)
        return frames_to_video