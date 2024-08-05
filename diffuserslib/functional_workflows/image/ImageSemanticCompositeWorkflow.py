from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.image.process import ImageCompositeNode, SemanticSegmentationNode, MaskDilationNode
from diffuserslib.functional.nodes.image.transform import ResizeImageNode
from diffuserslib.functional.nodes.user import *
from PIL import Image


class ImageSemanticCompositeWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Processor - Semantic Composite", Image.Image, workflow=True, subworkflow=True, realtime=False)


    def build(self):
        size_input = SizeUserInputNode(value = (512, 512), name = "size")
        foreground_input = ImageUploadInputNode(display = "Foreground", name = "foreground")
        background_input = ImageUploadInputNode(display = "Background", name = "background")
        mask_labels_input = TextAreaLinesInputNode(name = "mask_labels")
        mask_dilation = IntUserInputNode(value = 5, name = "mask_dilation")
        mask_feather = IntUserInputNode(value = 5, name = "mask_feather")

        segmentation_mask = SemanticSegmentationNode(image = foreground_input, mask_labels = mask_labels_input)
        resize_mask = ResizeImageNode(image = segmentation_mask, size = size_input, 
                                      type = ResizeImageNode.ResizeType.FIT, name="resize_mask")
        dilate_mask = MaskDilationNode(mask = resize_mask, dilation = mask_dilation, feather = mask_feather, name = "dilate_mask")

        resize_foreground = ResizeImageNode(image = foreground_input, size = size_input, 
                                            type = ResizeImageNode.ResizeType.FIT, name="resize_foreground")
        resize_background = ResizeImageNode(image = background_input, size = size_input, 
                                            type = ResizeImageNode.ResizeType.STRETCH, name="resize_background")
        
        composite = ImageCompositeNode(foreground = resize_foreground, background = resize_background, mask = dilate_mask)
        return composite
    