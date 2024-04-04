from diffuserslib.functional import *
from diffuserslib.functional.nodes import *


class AudioGenerationTortoiseWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Audio Generation - Tortoise", Audio, workflow=True, subworkflow=True)


    def build(self):
        files_input = FileSelectInputNode(filetype = "audio", name = "samples")
        prompt_input = TextAreaInputNode(value = "Hello World!", name = "prompt")

        samples_input = LoadAudioFilesNode(files = files_input, sample_rate = 22050, mono = True, name = "load_audio")
        audio = AudioGenerationTortoiseNode(prompt = prompt_input, samples = samples_input)
        return audio
    