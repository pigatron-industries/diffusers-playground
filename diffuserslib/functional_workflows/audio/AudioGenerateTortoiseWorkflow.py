from diffuserslib.functional import *
from diffuserslib.functional.nodes import *


class AudioGenerationTortoiseWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Audio Generation - Tortoise", Audio, workflow=True, subworkflow=True)


    def build(self):
        samples_input = FileSelectInputNode(filetype = "audio", name = "samples")
        prompt_input = TextAreaInputNode(value = "Hello World!", name = "prompt")
        audio = AudioGenerationTortoiseNode(prompt = prompt_input, samples = samples_input)
        return audio
    