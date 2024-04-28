from email.mime import image
from numpy import diff, size
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.image.diffusers.user import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.image.process import MLSDStraightLineDetectionNode
from diffuserslib.functional.nodes.image.process import SemanticSegmentationNode
from diffuserslib.functional.nodes.image.transform import ResizeImageNode
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.animated import FeedbackNode
from PIL import Image

class ImageDiffusionRoomDecorWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Diffusion - Conditioning - Room Decor", Image.Image, workflow=True, subworkflow=False)


    def build(self):
        image_input = ImageUploadInputNode(name = "image")
        size_input = SizeUserInputNode(value = (512, 512), name = "size")
        model_input = DiffusionModelUserInputNode(name = "model", basemodels=['sd_1_5'])
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
        image_resize = ResizeImageNode(image = image_input, size = size_input, type = ResizeImageNode.ResizeType.STRETCH)
        initimage = ConditioningInputNode(image = image_resize, model = 'initimage', scale = 0.85)
        mlsd = MLSDStraightLineDetectionNode(image = image_resize, name = "mlsd")
        mlsd_resize = ResizeImageNode(image = mlsd, size = size_input, type = ResizeImageNode.ResizeType.STRETCH)
        mlsd_conditioning = ConditioningInputNode(image = mlsd_resize, model = 'lllyasviel/control_v11p_sd15_mlsd', scale = 0.85)
        segmentation = SemanticSegmentationNode(image = image_resize, name = "segmentation")
        segmentation_resize = ResizeImageNode(image = segmentation, size = size_input, type = ResizeImageNode.ResizeType.STRETCH)
        segmentation_conditioning = ConditioningInputNode(image = segmentation_resize, model = 'lllyasviel/control_v11p_sd15_seg', scale = 0.85)

        diffusion = ImageDiffusionNode(models = model_input,
                                    loras = lora_input,
                                    # size = size_input,
                                    prompt = prompt_processor,
                                    negprompt = negprompt_input,
                                    steps = steps_input,
                                    cfgscale = cfgscale_input,
                                    seed = seed_input,
                                    scheduler = scheduler_input,
                                    conditioning_inputs = [initimage, mlsd_conditioning, segmentation_conditioning])
        
        feedback_image = FeedbackNode(type = Image.Image, input = diffusion, init_value = None, name = "feedback_image", display_name="Feedback Image")
        return diffusion, feedback_image
