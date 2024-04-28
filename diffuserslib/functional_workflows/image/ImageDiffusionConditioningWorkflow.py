from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.image.diffusers.user import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.image.transform import ResizeImageNode
from diffuserslib.functional.nodes.animated import FeedbackNode
from PIL import Image

class ImageDiffusionConditioningWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Diffusion - Conditioning", Image.Image, workflow=True, subworkflow=False)
        self.i = 0


    def build(self):
        model_input = DiffusionModelUserInputNode(name = "model")
        lora_input = LORAModelUserInputNode(diffusion_model_input = model_input, name = "lora")
        size_input = SizeUserInputNode(value = (512, 512), name = "size")
        prompt_input = TextAreaInputNode(value = "", name = "prompt")
        negprompt_input = TextAreaInputNode(value = "", name = "negprompt")
        steps_input = IntUserInputNode(value = 40, name = "steps")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        seed_input = SeedUserInputNode(value = None, name = "seed")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", 
                                                name = "scheduler",
                                                options = ImageDiffusionNode.SCHEDULERS)
        clipskip_input = IntUserInputNode(value = None, name = "clipskip")
        
        prompt_processor = RandomPromptProcessorNode(prompt = prompt_input, name = "prompt_processor")
        model_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(model_input.basemodel)))

        def create_conditioning_input():
            print("Creating conditioning input")
            i = self.i
            self.i += 1
            conditioning_model_input = ConditioningModelUserInputNode(diffusion_model_input = model_input, name = f"model{i}")
            scale_input = FloatUserInputNode(value = 1.0, name = f"scale{i}")
            image_input = ImageUploadInputNode(name = f"image_input{i}")
            resize_type_input = EnumSelectUserInputNode(value = ResizeImageNode.ResizeType.STRETCH, enum = ResizeImageNode.ResizeType, name = f"resize_type{i}")
            resize_image = ResizeImageNode(image = image_input, size = size_input, type = resize_type_input, name = f"resize_image{i}")
            return ConditioningInputNode(image = resize_image, model = conditioning_model_input, scale = scale_input, name = f"conditioning_input{i}")

        conditioning_inputs = ListUserInputNode(input_node_generator = create_conditioning_input, name = "conditioning_inputs")
        diffusion = ImageDiffusionNode(models = model_input,
                                    loras = lora_input,
                                    size = size_input,
                                    prompt = prompt_processor,
                                    negprompt = negprompt_input,
                                    steps = steps_input,
                                    cfgscale = cfgscale_input,
                                    seed = seed_input,
                                    scheduler = scheduler_input,
                                    clipskip = clipskip_input,
                                    conditioning_inputs = conditioning_inputs)
        
        feedback_image = FeedbackNode(type = Image.Image, input = diffusion, init_value = None, name = "feedback_image", display_name="Feedback Image")
        return diffusion, feedback_image
