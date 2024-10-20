from diffuserslib.functional import *
from diffuserslib.functional.nodes import *
from diffuserslib.functional.nodes.audio.StableAudioNode import StableAudioNode


class AudioGenerationStableAudioWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Audio Generation - Stable Audio", Audio, workflow=True, subworkflow=True)


    def build(self):
        audio_input = AudioUploadInputNode(mandatory=False, sample_rate = None, mono = False, name = "init_audio")
        noise_level_input = FloatUserInputNode(value = 1.0, name = "noise_level")
        prompt_input = TextAreaInputNode(value = "Hello World!", name = "prompt")
        negative_prompt_input = StringUserInputNode(value = "", name = "negative_prompt")
        duration_input = FloatUserInputNode(value = 10.0, name = "duration")
        steps_input = IntUserInputNode(value = 100, name = "steps")
        cfg_scale_input = FloatUserInputNode(value = 7, name = "cfg_scale")
        seed_input = SeedUserInputNode(value = None, name = "seed")

        audio = StableAudioNode(prompt = prompt_input, negative_prompt = negative_prompt_input, duration = duration_input, steps = steps_input, 
                                cfg_scale = cfg_scale_input, seed = seed_input, initaudio = audio_input, noise_level = noise_level_input)
        return audio
    