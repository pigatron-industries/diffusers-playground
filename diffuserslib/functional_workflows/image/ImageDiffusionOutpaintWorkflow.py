from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.image.process import *


class ImageDiffusionOutpaintWorkflow(WorkflowBuilder):
    """
    Outpaint an image: place the input image centred on a larger canvas of the chosen
    final size, then use an inpaint model to fill in the outer gaps. The final image is
    composited back with the original so the source pixels stay sharp and only the newly
    generated border is blended in.
    """

    def __init__(self):
        super().__init__("Image Diffusion - Outpaint", Image.Image, workflow=True, subworkflow=True)

    def build(self):
        # Outpaint inputs
        image_input = ImageUploadInputNode()
        size_input = SizeUserInputNode(value = (1024, 1024), name = "final_size")
        inpaint_model_input = DiffusionModelUserInputNode(modeltype = "inpaint", name="inpaint_model")
        loras_input = LORAModelUserInputNode(diffusion_model_input = inpaint_model_input, name = "lora")
        prompt_input = TextAreaInputNode(value = "", name="prompt")
        negprompt_input = StringUserInputNode(value = "", name="negprompt")
        seed_input = SeedUserInputNode(value = None, name="seed")
        steps_input = IntUserInputNode(value = 20, name = "steps")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", options = ImageDiffusionNode.SCHEDULERS, name="scheduler")

        # Place the original centred on a transparent canvas of the final size
        canvas = ImageCanvasResizeNode(image = image_input, size = size_input, name = "canvas_resize")

        # Mask: white (to fill) everywhere outside the original, black inside
        maskimage = ImageAlphaToMaskNode(image = canvas, smooth = False, name = "maskimage")

        # Conditioning images for the inpaint pass
        init_condition = ConditioningInputNode(image = canvas, model = "initimage", name = "outpaint_init_condition")
        mask_condition = ConditioningInputNode(image = maskimage, model = "maskimage", name = "outpaint_mask_condition")

        tilesize_calc = TileSizeCalculatorNode(image = canvas, name = "tile_size")

        # Inpaint diffusion over the final-size canvas
        prompt_processor = RandomPromptProcessorNode(prompt = prompt_input, name = "prompt_processor")
        inpaint_model_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(inpaint_model_input.basemodel)))
        image_diffusion = ImageDiffusionTiledNode(models = inpaint_model_input,
                                    loras = loras_input,
                                    prompt = prompt_processor,
                                    negprompt = negprompt_input,
                                    steps = steps_input,
                                    cfgscale = cfgscale_input,
                                    seed = seed_input,
                                    scheduler = scheduler_input,
                                    tilesize = tilesize_calc,
                                    conditioning_inputs = [ init_condition, mask_condition ], name="outpaint_diffusion")

        # Composite the generated result over the original, keeping the source sharp
        mask_dilate = MaskDilationNode(mask = maskimage, dilation = 10, feather = 10, name = "outpaint_mask_dilate")
        image_composite = ImageCompositeNode(foreground = image_diffusion, background = canvas, mask = mask_dilate, name = "composite")

        return image_composite
