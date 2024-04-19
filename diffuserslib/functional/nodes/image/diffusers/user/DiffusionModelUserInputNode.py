from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import ModelParameters
from diffuserslib.functional.nodes.image.diffusers.ImageDiffusionNode import ModelsType
from diffuserslib.functional.nodes.image.diffusers.RandomPromptProcessorNode import RandomPromptProcessorNode
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.nodes.user.UserInputNode import UserInputNode
from nicegui import ui


class DiffusionModelUserInputNode(UserInputNode):
    DEFAULT_BASEMODELS = ["sd_1_5", "sd_2_1", "sdxl_1_0", "sc_1_0"]

    def __init__(self, name:str="diffusion_model_user_input",
                 basemodels:List[str] = DEFAULT_BASEMODELS,
                 modeltype:str="generate"):
        self.basemodels = basemodels
        self.modeltype = modeltype
        self.basemodel = None
        self.model = None
        self.update_listeners = []
        self.selected_modifier = None
        super().__init__(name)


    def addUpdateListener(self, listener:Callable[[], None]):
        self.update_listeners.append(listener)


    def getValue(self) -> Tuple[str|None, str|None]:
        return (self.basemodel, self.model)
    

    def setValue(self, value:Tuple[str|None, str|None]):
        self.basemodel = value[0]
        self.model = value[1]
        for listener in self.update_listeners:
            listener()


    @ui.refreshable
    def gui(self):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines not initialised")  
        with ui.column().classes('grow'):
            models = DiffusersPipelines.pipelines.presets.getModelsByTypeAndBase(self.modeltype, self.basemodel)
            with ui.row().classes('w-full'):
                self.basemodel_dropdown = ui.select(options=self.basemodels, value=self.basemodel, label="Base Model", on_change=lambda e: self.updateModels()).bind_value(self, 'basemodel').classes('grow')  
                ui.button(icon="settings", on_click=lambda e: self.modelSettings()).classes('align-middle').props('dense')
            self.model_dropdown = ui.select(options=sorted(list(models.keys())), label="Model").bind_value(self, 'model').classes('w-full')


    def modelSettings(self):
        with ui.dialog(value = True):
            with ui.row().style('height: 100%; max-width: 1000px;'):
                self.embeddingsList()
                self.modifiersList()   # type: ignore
                self.modifierEditor()  # type: ignore
        


    def embeddingsList(self):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines not initialised") 
        with ui.card().style('height:100%;width:300px;'):
            with ui.column().classes('grow').style('width:100%;'):
                ui.label("Embeddings")
                embeddings = DiffusersPipelines.pipelines.getEmbeddingTokens(self.basemodel)
                with ui.scroll_area().classes('w-32 h-32 border grow').style('width:100%;'):
                    with ui.list().props('dense'):
                        for embedding in embeddings:
                            with ui.item():
                                ui.label(embedding)


    @ui.refreshable
    def modifiersList(self):
        with ui.card().style('height:100%;width:300px;'):
            with ui.column().classes('grow').style('width:100%;'):
                ui.label("Modifiers")
                modifiers = RandomPromptProcessorNode.modifier_dict
                with ui.scroll_area().classes('w-32 h-32 border grow').style('width:100%;'):
                    with ui.list().props('dense'):
                        for modifier in modifiers:
                            self.modifierListItem(modifier)
                

    def modifierListItem(self, modifier:str):
        if(self.selected_modifier == modifier):
            style = 'background-color:rgb(88, 152, 212)'
        else:
            style = ''
        with ui.item(on_click=lambda: self.selectModifier(modifier)).style(style):
            if(self.selected_modifier == modifier):
                ui.label(modifier)
            else:
                ui.label(modifier)


    @ui.refreshable
    def modifierEditor(self):
        with ui.card().style('height:100%;width:300px;'):
            if(self.selected_modifier is not None):
                with ui.column().classes('grow').style('width:100%;'):
                    ui.label("Edit Modifier")
                    modifierItems = RandomPromptProcessorNode.modifier_dict[self.selected_modifier]
                    with ui.scroll_area().classes('w-32 h-32 border grow').style('width:100%;'):
                        with ui.list().props('dense'):
                            for modifierItem in modifierItems:
                                self.modifierItem(modifierItem)


    def modifierItem(self, modifierItem:str):
        with ui.item():
            ui.label(modifierItem)


    def selectModifier(self, modifier:str):
        self.selected_modifier = modifier
        self.modifiersList.refresh()
        self.modifierEditor.refresh()
        


    def updateModels(self):
        print("updateModels")
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines not initialised")  
        if(self.basemodel is not None):
            models = DiffusersPipelines.pipelines.presets.getModelsByTypeAndBase("generate", self.basemodel)
            self.model = None
            self.model_dropdown.options = list(models.keys())
            self.gui.refresh()
            for listener in self.update_listeners:
                listener()


    def process(self) -> ModelsType:
        if(self.model is None):
            raise Exception("Model not selected")

        modelparams:List[ModelParameters] = []
        modelparams.append(ModelParameters(self.model, 1.0))
        return modelparams