from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from llama_index.multi_modal_llms.ollama import OllamaMultiModal
import io


class ImageLanguageModelCompletionNode(FunctionalNode):
    def __init__(self, 
                 prompt:StringFuncType,
                 images: ImagesFuncType,
                 model:StringFuncType = "llava:13b",
                 name:str = "llm_completion"):
        super().__init__(name)
        self.addParam("model", model, str)
        self.addParam("images", images, List[Image.Image])
        self.addParam("prompt", prompt, str)
        self.model = None
        self.text = ""


    def process(self, model:str, images:List[Image.Image], prompt:str) -> str:
        if(model != self.model):
            self.model = model
            self.llm = OllamaMultiModal(model = model, request_timeout = 120)
        image_documents = []
        for image in images:
            binary_stream = io.BytesIO()
            image.save(binary_stream, format='PNG')
            image_documents.append(binary_stream.getvalue())
        response = self.llm.stream_complete(prompt, image_documents)
        self.text = ""
        for r in response:
            self.text = r.text
        return self.text
    
    
    def getProgress(self) -> WorkflowProgress|None:
        return WorkflowProgress(0, self.text)
        
