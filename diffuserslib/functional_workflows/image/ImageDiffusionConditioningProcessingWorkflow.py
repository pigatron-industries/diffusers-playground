from email.mime import image
from numpy import diff, size
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.image.diffusers.user import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.image.process import ControlNetProcessorNode
from diffuserslib.functional.nodes.image.process import SemanticSegmentationNode
from diffuserslib.functional.nodes.image.transform import ResizeImageNode
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.animated import FeedbackNode
from PIL import Image

class ImageDiffusionConditioningProcessingWorkflow(WorkflowBuilder):

    """
    """

    def __init__(self):
        super().__init__("Image Diffusion - Conditioning with Processing", Image.Image, workflow=True, subworkflow=True)
        self.i = 0


    def build(self):
        image_input = ImageUploadInputNode(name = "image")
        size_input = SizeUserInputNode(value = (512, 512), name = "size")
        model_input = DiffusionModelUserInputNode(name = "model", basemodels=['sd_1_5', 'sdxl_1_0'])
        lora_input = LORAModelUserInputNode(diffusion_model_input = model_input, name = "lora")
        prompt_input = TextAreaInputNode(value = "", name = "prompt")
        negprompt_input = TextAreaInputNode(value = "", name = "negprompt")
        steps_input = IntUserInputNode(value = 40, name = "steps")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        seed_input = SeedUserInputNode(value = None, name = "seed")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", 
                                                name = "scheduler",
                                                options = ImageDiffusionNode.SCHEDULERS)
        
        prompt_processor = RandomPromptProcessorNode(prompt = prompt_input, name = "prompt_processor")
        model_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(model_input.basemodel)))

        # image conditioning
        image_resize = ResizeImageNode(image = image_input, size = size_input, type = ResizeImageNode.ResizeType.FIT)
        # initimage_scale_input = FloatUserInputNode(value = 0.85, name = "initimage_scale")
        # initimage = ConditioningInputNode(image = image_resize, model = 'initimage', scale = initimage_scale_input)

        def create_conditioning_input():
            print("Creating conditioning input")
            i = self.i
            self.i += 1
            conditioning_model_input = ConditioningModelUserInputNode(diffusion_model_input = model_input, name = f"model{i}")
            scale_input = FloatUserInputNode(value = 1.0, name = f"scale{i}")
            processor_input = ListSelectUserInputNode(name = "processor", value="canny", options = ControlNetProcessorNode.PROCESSORS)

            controlnet_processor = ControlNetProcessorNode(image = image_resize, processor = processor_input, name = f"controlnet_processor{i}")
            controlnet_resize_image = ResizeImageNode(image = controlnet_processor, size = size_input, type = ResizeImageNode.ResizeType.STRETCH, name = f"resize_image{i}")
            return ConditioningInputNode(image = controlnet_resize_image, model = conditioning_model_input, scale = scale_input, name = f"conditioning_input")

        conditioning_inputs = ListUserInputNode(input_node_generator = create_conditioning_input, name = "conditioning_inputs")
        diffusion = ImageDiffusionNode(models = model_input,
                                    loras = lora_input,
                                    # size = size_input,
                                    prompt = prompt_processor,
                                    negprompt = negprompt_input,
                                    steps = steps_input,
                                    cfgscale = cfgscale_input,
                                    seed = seed_input,
                                    scheduler = scheduler_input,
                                    conditioning_inputs = conditioning_inputs)
        
        feedback_image = FeedbackNode(type = Image.Image, input = diffusion, init_value = None, name = "feedback_image", display_name="Feedback Image")
        return diffusion, feedback_image
