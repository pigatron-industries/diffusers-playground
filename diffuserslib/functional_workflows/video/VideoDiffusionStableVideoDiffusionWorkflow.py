from diffuserslib.functional import *


class VideoDiffusionStableVideoDiffussionWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video Diffusion - Stable Video Diffusion", Video, workflow=True, subworkflow=True)

    def build(self):
        model_input = DiffusionModelUserInputNode(basemodels=["svd_1_0"], name="model")
        size_input = SizeUserInputNode(value = (512, 512), name="size")
        seed_input = SeedUserInputNode(value = None, name="seed")
        frames_input = IntUserInputNode(value = 16, name = "frames")
        fps_input = IntUserInputNode(value = 7, name = "fps")

        image_input = ImageSelectInputNode(name = "image")

        resize_image = ResizeImageNode(image = image_input, size = size_input, type = ResizeImageNode.ResizeType.STRETCH)
        conditioning_inputs = [ConditioningInputNode(image = resize_image, model = "initimage")]

        animatediff = VideoDiffusionStableVideoDiffusionNode(models = model_input, 
                                                            size = size_input, 
                                                            seed = seed_input,
                                                            frames = frames_input,
                                                            conditioning_inputs = conditioning_inputs)
        frames_to_video = FramesToVideoNode(frames = animatediff, fps = fps_input)
        
        # model_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(models_input.basemodel)))
        return frames_to_video