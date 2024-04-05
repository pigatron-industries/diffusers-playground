from diffuserslib.functional import *
from diffuserslib.functional.nodes import *


class AudioGenerationBarkWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Audio Generation - Bark", Audio, workflow=True, subworkflow=True)


    def build(self):
        files_input = FileSelectInputNode(filetype = "audio", name = "samples")
        prompt_input = TextAreaInputNode(value = "Hello World!", name = "prompt")

        samples_input = LoadAudioFilesNode(files = files_input, name = "load_audio")
        audio = AudioGenerationBarkNode(prompt = prompt_input, sample = samples_input, name = "audio_bark")
        return audio
    