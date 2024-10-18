from diffuserslib.functional import *
from diffuserslib.functional.nodes import *
from diffuserslib.functional.nodes.audio.StableAudioNode import StableAudioNode


class AudioTestWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Audio Generation - Generic", Audio, workflow=True, subworkflow=True)


    def build(self):
        audio_input = AudioUploadInputNode(mandatory=False, sample_rate = None, mono = False, name = "audio_input")
        output = NoOpNode(input = audio_input, name = "placeholder")
        return output
    