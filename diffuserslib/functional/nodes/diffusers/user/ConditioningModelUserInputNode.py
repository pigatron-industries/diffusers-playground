from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import ControlImageType
from diffuserslib.functional.nodes.diffusers.ImageDiffusionNode import ModelsType
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from .DiffusionModelUserInputNode import DiffusionModelUserInputNode
from ...user.UserInputNode import UserInputNode
from nicegui import ui


class ConditioningModelUserInputNode(UserInputNode):

    def __init__(self, diffusion_model_input:DiffusionModelUserInputNode, name:str="conditioning_model_user_input"):
        self.model = None
        self.diffusion_model_input = diffusion_model_input
        diffusion_model_input.addUpdateListener(self.updateModels)
        super().__init__(name)


    def getValue(self) -> str|None:
        return self.model
    

    def setValue(self, value:str|None):
        self.model = value


    @ui.refreshable
    def gui(self):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines not initialised")  
        models = DiffusersPipelines.pipelines.presets.getModelsByTypeAndBase("conditioning", self.diffusion_model_input.basemodel)
        options = [ControlImageType.IMAGETYPE_INITIMAGE]+list(models.keys())
        if(self.model not in options):
            self.model = None
        self.model_dropdown = ui.select(options=[ControlImageType.IMAGETYPE_INITIMAGE]+list(models.keys()), value=self.model, label="Model").bind_value(self, 'model').classes('grow')


    def updateModels(self):
        self.gui.refresh()


    def process(self) -> str:
        if(self.model is None):
            raise Exception("Model not selected")
        return self.model