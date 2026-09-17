from diffuserslib.functional import *
from diffuserslib.functional.nodes import *
from diffuserslib.functional.nodes.video.MinimaxH3VideoNode import MinimaxH3VideoNode


class MinimaxH3VideoWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video Diffusion - MiniMax H3", Video, workflow=True, subworkflow=False, realtime=False)


    def build(self):
        prompt_input = TextAreaInputNode(value="", name="prompt")
        first_image_input = ImageUploadInputNode(mandatory=False, display="First frame", name="first_image")
        last_image_input = ImageUploadInputNode(mandatory=False, display="Last frame", name="last_image")
        size_input = SizeUserInputNode(value=(768, 448), name="resolution")
        duration_input = FloatUserInputNode(value=5.0, name="duration")
        steps_input = IntUserInputNode(value=9, name="steps")
        seed_input = SeedUserInputNode(value=None, name="seed")

        output_video = MinimaxH3VideoNode(prompt=prompt_input, first_image=first_image_input, last_image=last_image_input,
                                  resolution=size_input, duration=duration_input, steps=steps_input, seed=seed_input)
        
        return output_video
