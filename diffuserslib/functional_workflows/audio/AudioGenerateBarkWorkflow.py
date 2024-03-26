from diffuserslib.functional import *
from diffuserslib.functional.nodes import *


class AudioGenerationBarkWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Audio Generation - Bark", Image.Image, workflow=True, subworkflow=True)


    def build(self):
        prompt_input = TextAreaInputNode(value = "Hello World!", name = "prompt")
        audio = AudioGenerationBarkNode(prompt = prompt_input)
        return audio
    