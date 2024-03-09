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
        self.weight = 1.0
        self.diffusion_model_input = diffusion_model_input
        diffusion_model_input.addUpdateListener(self.updateModels)
        super().__init__(name)


    def getValue(self) -> tuple[str|None, float]|None:
        return self.lora, self.weight
    

    def setValue(self, value:tuple[str|None, float]|None):
        if value is not None:
            self.lora = value[0]
            self.weight = value[1]
        else:
            self.lora = None
            self.weight = 1.0


    @ui.refreshable
    def gui(self):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines not initialised")  
        loras = DiffusersPipelines.pipelines.getLORAsByBase(self.diffusion_model_input.basemodel)
        if (self.lora is not None and self.lora not in loras):
            self.lora = None
        self.lora_dropdown = ui.select(options=[None]+list(loras), value=self.lora, label="Lora").bind_value(self, 'lora').classes('grow')
        self.weight_dropdown = ui.number(value=1.0, label="Weight").bind_value(self, 'weight')


    def updateModels(self):
        self.gui.refresh()


    def process(self) -> LorasType:
        loraparams:List[LoraParameters] = []
        if self.lora is not None:
            loraparams.append(LoraParameters(self.lora, self.weight))
        return loraparams