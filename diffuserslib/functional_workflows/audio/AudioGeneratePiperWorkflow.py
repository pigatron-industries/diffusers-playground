from diffuserslib.functional import *
from diffuserslib.functional.nodes.audio.speech.TextToSpeechPiperNode import *
from diffuserslib.functional.nodes.user.UserInputNode import *


class AudioGenerationPiperWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Audio Generation - Piper Text to Speech", Audio, workflow=True, subworkflow=True)


    def build(self):
        models = GlobalConfig.getModelsByBase("tts", "piper_1_0")
        model_dict = {model['name']: model['id'] for model in models}

        model_input = DictSelectUserInputNode(value = "", options = model_dict, name = "model")
        speaker_input = IntUserInputNode(value = 0, name = "speaker_id")
        prompt_input = TextAreaInputNode(value = "Hello World!", name = "prompt")
        length_scale_input = FloatUserInputNode(value = 1.0, name = "length_scale")
        noise_scale_input = FloatUserInputNode(value = None, name = "noise_scale")
        noise_w_input = FloatUserInputNode(value = None, name = "noise_w")

        audio = TextToSpeechPiperNode(prompt = prompt_input, model = model_input, speakerid = speaker_input, 
                                      length_scale = length_scale_input, noise_scale = noise_scale_input, noise_w = noise_w_input)
        return audio
    