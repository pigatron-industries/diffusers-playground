from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import LoraParameters
from diffuserslib.functional.nodes.image.diffusers.ImageDiffusionNode import LorasType
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.nodes.user.UserInputNode import UserInputNode
from .DiffusionModelUserInputNode import DiffusionModelUserInputNode
from nicegui import ui


class LORAModelUserInputNode(UserInputNode):

    def __init__(self, diffusion_model_input:DiffusionModelUserInputNode, name:str="lora_model_user_input"):
        self.selected_loras = []
        self.selected_weights = []
        self.diffusion_model_input = diffusion_model_input
        diffusion_model_input.addUpdateListener(self.updateModels)
        super().__init__(name)


    def getValue(self) -> List[Tuple[str|None, float]]|None:
        return list(zip(self.selected_loras, self.selected_weights))
    

    def setValue(self, value:List[Tuple[str|None, float]]|None):
        try:
            if value is not None:
                self.selected_loras = [v[0] for v in value]
                self.selected_weights = [v[1] for v in value]
        except:
            self.selected_loras = []
            self.selected_weights = []


    @ui.refreshable
    def gui(self):
        ui.label(f"LORAs")
        for i in range(len(self.selected_loras)):
            self.loraModelGui(i)
        with ui.row().classes('w-full'):
            ui.label().classes('w-8')
            ui.button(icon="add", on_click = lambda e: self.addInput(0)).props('dense').classes('align-middle')


    def loraModelGui(self, i):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines not initialised")  
        loras = DiffusersPipelines.pipelines.getLORAsByBase(self.diffusion_model_input.basemodel)

        selected_lora = self.selected_loras[i]
        if selected_lora not in loras:
            selected_lora = None
        
        with ui.row().classes('w-full'):
            ui.label().classes('w-8')
            ui.select(options=sorted(list(loras)), with_input=True, value=selected_lora, label="Lora", on_change=lambda e: self.updateLora(i, e.value)).classes('grow') 
            ui.number(value=self.selected_weights[i], label="Weight", on_change=lambda e: self.updateWeight(i, e.value)).classes('small-number')
            ui.button(icon="remove", on_click = lambda e: self.removeInput(i)).props('dense').classes('align-middle')


    def addInput(self, i):
        self.selected_loras.insert(i, "")
        self.selected_weights.insert(i, 1.0)
        self.gui.refresh()

    
    def removeInput(self, i):
        self.selected_loras.pop(i)
        self.selected_weights.pop(i)
        self.gui.refresh()


    def updateLora(self, i, value):
        self.selected_loras[i] = value
        

    def updateWeight(self, i, value):
        self.selected_weights[i] = value


    def updateModels(self):
        self.gui.refresh()


    def processValue(self) -> LorasType:
        loraparams:List[LoraParameters] = []
        for i in range(len(self.selected_loras)):
            if self.selected_loras[i] != "":
                loraparams.append(LoraParameters(self.selected_loras[i], self.selected_weights[i]))
        return loraparams