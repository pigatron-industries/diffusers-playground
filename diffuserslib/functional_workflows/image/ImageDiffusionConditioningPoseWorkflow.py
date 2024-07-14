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

class ImageDiffusionConditioningPoseWorkflow(WorkflowBuilder):

    """
    Image diffusion with a set of fixed conditioning models for keeping human pose and depth map of original image.
    Uses open pose and depth map control nets as conditioning.
    """

    def __init__(self):
        super().__init__("Image Diffusion - Conditioning - Pose", Image.Image, workflow=True, subworkflow=False)


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
        image_resize = ResizeImageNode(image = image_input, size = size_input, type = ResizeImageNode.ResizeType.STRETCH)
        initimage = ConditioningInputNode(image = image_resize, model = 'initimage', scale = 0.85)

        pose = ControlNetProcessorNode(image = image_resize, processor = "openpose", name = "pose")
        pose_resize = ResizeImageNode(image = pose, size = size_input, type = ResizeImageNode.ResizeType.STRETCH)
        pose_conditioning = ConditioningInputNode(image = pose_resize, model = 'xinsir/controlnet-openpose-sdxl-1.0', scale = 0.85)  # TODO select correct model for base

        depth = ControlNetProcessorNode(image = image_resize, processor = "depth_zoe", name = "depth")
        depth_resize = ResizeImageNode(image = depth, size = size_input, type = ResizeImageNode.ResizeType.STRETCH)
        depth_conditioning = ConditioningInputNode(image = depth_resize, model = 'diffusers/controlnet-depth-sdxl-1.0', scale = 0.85)   # TODO select correct model for base

        diffusion = ImageDiffusionNode(models = model_input,
                                    loras = lora_input,
                                    # size = size_input,
                                    prompt = prompt_processor,
                                    negprompt = negprompt_input,
                                    steps = steps_input,
                                    cfgscale = cfgscale_input,
                                    seed = seed_input,
                                    scheduler = scheduler_input,
                                    conditioning_inputs = [initimage, pose_conditioning, depth_conditioning])
        
        feedback_image = FeedbackNode(type = Image.Image, input = diffusion, init_value = None, name = "feedback_image", display_name="Feedback Image")
        return diffusion, feedback_image
