from diffuserslib.functional import *
from diffuserslib.functional.nodes import *
from diffuserslib.functional.nodes.audio.StableAudioNode import StableAudioNode


class AudioGenerationStableAudioWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Audio Generation - Stable Audio 3", Audio, workflow=True, subworkflow=True)


    def build(self):
        audio_input = AudioUploadInputNode(mandatory=False, sample_rate = None, mono = False, name = "init_audio")
        noise_level_input = FloatUserInputNode(value = 1.0, name = "noise_level")
        prompt_input = TextAreaInputNode(value = "A gentle piano melody with soft strings in a concert hall", name = "prompt")
        negative_prompt_input = StringUserInputNode(value = "", name = "negative_prompt")
        duration_input = FloatUserInputNode(value = 10.0, name = "duration")
        # 0 = let the model choose its native step count (8 distilled / ~100 base)
        steps_input = IntUserInputNode(value = 0, name = "steps")
        cfg_scale_input = FloatUserInputNode(value = 1.0, name = "cfg_scale")
        seed_input = SeedUserInputNode(value = None, name = "seed")
        model_input = StringUserInputNode(value = "stabilityai/stable-audio-3-medium", name = "model")

        audio = StableAudioNode(prompt = prompt_input, negative_prompt = negative_prompt_input, duration = duration_input, steps = steps_input,
                                cfg_scale = cfg_scale_input, seed = seed_input, initaudio = audio_input, noise_level = noise_level_input,
                                model = model_input)
        return audio
