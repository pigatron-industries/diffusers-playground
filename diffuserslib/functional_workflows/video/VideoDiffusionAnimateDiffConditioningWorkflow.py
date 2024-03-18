from diffuserslib.functional import *


class VideoDiffusionAnimateDiffConditioningWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video Diffusion - AnimateDiff Conditioning", Video, workflow=True, subworkflow=False)

    def build(self):
        models_input = DiffusionModelUserInputNode(basemodels=["sd_1_5"])
        size_input = SizeUserInputNode(value = (512, 512))
        prompt_input = TextAreaInputNode(value = "", name="prompt")
        negprompt_input = StringUserInputNode(value = "", name="negprompt")
        seed_input = SeedUserInputNode(value = None, name="seed")
        steps_input = IntUserInputNode(value = 20, name = "steps")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", options = ImageDiffusionNode.SCHEDULERS, name="scheduler")
        num_frames_input = IntUserInputNode(value = 16, name = "frames")

        prompt_processor = RandomPromptProcessorNode(prompt = prompt_input, name = "prompt_processor")
        models_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(models_input.basemodel)))

        def create_conditioning_input():
            conditioning_model_input = ConditioningModelUserInputNode(diffusion_model_input = models_input, name = "model")
            scale_input = FloatUserInputNode(value = 1.0, name = "scale")
            frames_input = VideoUploadInputNode(name = "frames_input")
            # resize_type_input = EnumSelectUserInputNode(value = ResizeImageNode.ResizeType.EXTEND, enum = ResizeImageNode.ResizeType, name = "resize_type")
            # resize_image = ResizeImageNode(image = image_input, size = size_input, type = resize_type_input, name = "resize_image")
            return FramesConditioningInputNode(image = frames_input, model = conditioning_model_input, scale = scale_input, name = "conditioning_input")

        conditioning_inputs = ListUserInputNode(input_node_generator = create_conditioning_input, name = "conditioning_inputs")
        animatediff = VideoDiffusionAnimateDiffNode(models = models_input, 
                                                    size = size_input, 
                                                    prompt = prompt_processor,
                                                    negprompt = negprompt_input,
                                                    seed = seed_input,
                                                    steps = steps_input,
                                                    cfgscale = cfgscale_input,
                                                    scheduler = scheduler_input,
                                                    frames = num_frames_input,
                                                    conditioning_inputs = conditioning_inputs)
        frames_to_video = FramesToVideoNode(frames = animatediff, fps = 7.5)
        
        return frames_to_video