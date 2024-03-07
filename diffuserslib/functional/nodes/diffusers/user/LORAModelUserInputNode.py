from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import LoraParameters
from diffuserslib.functional.nodes.diffusers.ImageDiffusionNode import LorasType
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from .DiffusionModelUserInputNode import DiffusionModelUserInputNode
from ...user.UserInputNode import UserInputNode
from nicegui import ui


class LORAModelUserInputNode(UserInputNode):

    def __init__(self, diffusion_model_input:DiffusionModelUserInputNode, name:str="lora_model_user_input"):
        self.lora = None
        self.diffusion_model_input = diffusion_model_input
        diffusion_model_input.addUpdateListener(self.updateModels)
        super().__init__(name)


    def getValue(self) -> str|None:
        return self.lora
    

    def setValue(self, value:str|None):
        self.lora = value


    @ui.refreshable
    def gui(self):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines not initialised")  
        loras = DiffusersPipelines.pipelines.getLORAsByBase(self.diffusion_model_input.basemodel)
        if (self.lora is not None and self.lora not in loras):
            self.lora = None
        self.lora_dropdown = ui.select(options=[None]+list(loras), value=self.lora, label="Lora").bind_value(self, 'lora').classes('grow')


    def updateModels(self):
        self.gui.refresh()


    def process(self) -> LorasType:
        loraparams:List[LoraParameters] = []
        if self.lora is not None:
            loraparams.append(LoraParameters(self.lora, 1.0))
        return loraparams