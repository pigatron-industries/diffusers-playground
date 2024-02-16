from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import ModelParameters
from diffuserslib.functional.nodes.diffusers.ImageDiffusionNode import ModelsType
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from .UserInputNode import UserInputNode
from nicegui import ui


class DiffusionModelUserInputNode(UserInputNode):
    basemodels = ["sd_1_5", "sd_2_1", "sdxl_1_0"]

    def __init__(self, name:str="diffusion_model_user_input"):
        self.basemodel = None
        self.model = None
        self.update_listeners = []
        super().__init__(name)


    def addUpdateListener(self, listener:Callable[[], None]):
        self.update_listeners.append(listener)


    def getValue(self) -> Tuple[str|None, str|None]:
        return (self.basemodel, self.model)
    

    def setValue(self, value:Tuple[str|None, str|None]):
        self.basemodel = value[0]
        self.model = value[1]


    @ui.refreshable
    def ui(self):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines not initialised")  
        with ui.column().classes('grow'):
            models = DiffusersPipelines.pipelines.presets.getModelsByTypeAndBase("generate", self.basemodel)
            self.basemodel_dropdown = ui.select(options=self.basemodels, value=self.basemodel, label="Base Model", on_change=lambda e: self.updateModels()).bind_value(self, 'basemodel').classes('w-full')  
            self.model_dropdown = ui.select(options=list(models.keys()), value=self.model, label="Model").bind_value(self, 'model').classes('w-full')


    def updateModels(self):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines not initialised")  
        if(self.basemodel is not None):
            models = DiffusersPipelines.pipelines.presets.getModelsByTypeAndBase("generate", self.basemodel)
            self.model = None
            self.model_dropdown.options = list(models.keys())
            self.ui.refresh()
            for listener in self.update_listeners:
                listener()


    def process(self) -> ModelsType:
        if(self.model is None):
            raise Exception("Model not selected")

        modelparams:List[ModelParameters] = []
        modelparams.append(ModelParameters(self.model, 1.0))
        return modelparams