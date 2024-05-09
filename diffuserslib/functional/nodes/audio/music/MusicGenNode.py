from diffuserslib.functional.types import *
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from transformers import MusicgenForConditionalGeneration, AutoProcessor


class MusicGenNode(FunctionalNode):

    def __init__(self, 
                 prompt:StringFuncType = "",
                 name:str="musicgen"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.model = None
        self.processor = None
        
        
    def process(self, prompt:str):
        if (self.model is None):
            self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
            self.processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
        assert self.model is not None, "Model is not initialized"
        assert self.processor is not None, "Processor is not initialized"

        if (prompt is not None and len(prompt) > 0):
            inputs = self.processor(text = prompt, return_tensors="pt", padding=True)
            inputs['guidance_scale'] = 3
        else:
            inputs = self.model.get_unconditional_inputs(num_samples=1)
        
        audio_values = self.model.generate(**inputs, do_sample=True, max_new_tokens=512)

        sampling_rate = self.model.config.audio_encoder.sampling_rate
        audio_array = audio_values[0, 0].numpy()
        return Audio(audio_array, sampling_rate)
    