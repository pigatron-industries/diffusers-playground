from diffuserslib.functional import *
from diffuserslib.functional.nodes import *


class AudioGenerationBarkWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Audio Generation - Tortoise", Audio, workflow=True, subworkflow=True)


    def build(self):
        prompt_input = TextAreaInputNode(value = "Hello World!", name = "prompt")
        audio = AudioGenerationTortoiseNode(prompt = prompt_input)
        return audio
    