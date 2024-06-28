from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from controlnet_aux.processor import Processor


class ControlNetProcessorNode(FunctionalNode):

    PROCESSORS = ["canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
                  "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
                  "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
                  "scribble_hed", "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
                  "softedge_pidinet", "softedge_pidsafe", "dwpose"]

    def __init__(self, 
                 image:ImageFuncType, 
                 processor:StringFuncType,
                 name:str="controlnet_processor"):
        super().__init__(name)
        self.addParam("processor", processor, str)
        self.addParam("image", image, Image.Image)
        self.model = None
        
        
    def process(self, image:Image.Image, processor:str) -> Image.Image|None:
        if self.model is None:
            self.model = Processor(processor)
        if image is None:
            return None
        outimage = self.model(image, to_pil=True)
        assert isinstance(outimage, Image.Image)
        return outimage
    