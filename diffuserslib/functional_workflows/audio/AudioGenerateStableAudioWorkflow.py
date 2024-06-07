from diffuserslib.functional import *
from diffuserslib.functional.nodes import *
from diffuserslib.functional.nodes.audio.StableAudioNode import StableAudioNode


class AudioGenerationStableAudioWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Audio Generation - Stable Audio", Audio, workflow=True, subworkflow=True)


    def build(self):
        # files_input = FileSelectInputNode(filetype = "audio", name = "samples")
        prompt_input = TextAreaInputNode(value = "Hello World!", name = "prompt")
        duration_input = FloatUserInputNode(value = 10.0, name = "duration")
        audio = StableAudioNode(prompt = prompt_input, duration = duration_input)
        return audio
    