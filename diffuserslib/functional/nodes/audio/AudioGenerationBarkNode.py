from diffuserslib.functional.types import *
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from transformers import BarkModel, BarkProcessor
from transformers.models.bark.generation_configuration_bark import BarkSemanticGenerationConfig, BarkCoarseGenerationConfig, BarkFineGenerationConfig
import torch


class AudioGenerationBarkNode(FunctionalNode):

    def __init__(self, 
                 prompt:StringFuncType = "",
                 voice:StringFuncType = "v2/en_speaker_6",
                 name:str="audio_bark"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.addParam("voice", voice, str)
        self.processor = None
        self.model = None
        self.history_prompt = None
        
        
    def process(self, prompt:str, voice:str):
        # TODO history_prompt loaded from file as a voice preset
        audio_array, self.history_prompt = self.generate(prompt, self.history_prompt)

        audio_array = audio_array.detach().numpy().squeeze()
        return Audio(audio_array, self.model.generation_config.sample_rate)
    

    def generate(self, prompt:str, history_prompt = None):
        if (self.processor is None or self.model is None):
            self.processor = BarkProcessor.from_pretrained("suno/bark")
            self.model = BarkModel.from_pretrained("suno/bark")
        inputs = self.processor(prompt, voice_preset = history_prompt)
        audio_array = self.model.generate(**inputs)
        return audio_array, None


    # TODO This was an attempt to save the history_prompt from the previous generation in order to reuse it in the next generation
    # This issue is related https://github.com/huggingface/transformers/issues/28890

    # def generate(self, prompt:str, history_prompt = None):
    #     if (self.processor is None or self.model is None):
    #         self.processor = BarkProcessor.from_pretrained("suno/bark")
    #         self.model = BarkModel.from_pretrained("suno/bark")

    #     inputs = self.processor(prompt)  #voice_preset = 'v2/en_speaker_6'
    #     history_prompt = inputs["history_prompt"] if "history_prompt" in inputs else None
    #     # print(inputs["history_prompt"]["semantic_prompt"].shape)
    #     # print(inputs["history_prompt"]["coarse_prompt"].shape)
    #     # print(inputs["history_prompt"]["fine_prompt"].shape)

    #     semantic_generation_config = BarkSemanticGenerationConfig(**self.model.generation_config.semantic_config)
    #     coarse_generation_config = BarkCoarseGenerationConfig(**self.model.generation_config.coarse_acoustics_config)
    #     fine_generation_config = BarkFineGenerationConfig(**self.model.generation_config.fine_acoustics_config)
        
    #     # 1. Generate from the semantic model
    #     semantic_output = self.model.semantic.generate(
    #         inputs["input_ids"],
    #         history_prompt = history_prompt,
    #         semantic_generation_config = semantic_generation_config,
    #         attention_mask = inputs["attention_mask"]
    #     )

    #     print(semantic_output.shape)

    #     # 2. Generate from the coarse model
    #     coarse_output = self.model.coarse_acoustics.generate(
    #         semantic_output,
    #         history_prompt = history_prompt,
    #         semantic_generation_config = semantic_generation_config,
    #         coarse_generation_config = coarse_generation_config,
    #         codebook_size = self.model.generation_config.codebook_size
    #     )

    #     print(coarse_output.shape)

    #     # 3. Generate from the fine model
    #     fine_output = self.model.fine_acoustics.generate(
    #         coarse_output,
    #         history_prompt = history_prompt,
    #         semantic_generation_config = semantic_generation_config,
    #         coarse_generation_config = coarse_generation_config,
    #         fine_generation_config = fine_generation_config,
    #         codebook_size = self.model.generation_config.codebook_size,
    #     )

    #     print(fine_output.shape)

    #     history = {
    #         "semantic_prompt": semantic_output[0],
    #         "coarse_prompt": coarse_output[0],
    #         "fine_prompt": fine_output[0]
    #     }

    #     audio = self.model.codec_decode(fine_output)
    #     return audio, history

