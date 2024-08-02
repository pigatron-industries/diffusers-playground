from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import ModelParameters
from diffuserslib.functional.nodes.image.diffusers.ImageDiffusionNode import ModelsType
from diffuserslib.functional.nodes.image.diffusers.RandomPromptProcessorNode import RandomPromptProcessorNode
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.nodes.user.UserInputNode import UserInputNode
from nicegui import ui


class DiffusionModelUserInputNode(UserInputNode):
    DEFAULT_BASEMODELS = ["sd_1_5", "sd_2_1", "sdxl_1_0", "sc_1_0", "pixart_sigma", "sd_3_0", "auraflow", "kwai_kolors", "flux"]

    def __init__(self, name:str="diffusion_model_user_input",
                 basemodels:List[str] = DEFAULT_BASEMODELS,
                 modeltype:str="generate"):
        self.basemodels = basemodels
        self.modeltype = modeltype
        self.basemodel = None
        self.selected_models:List[str] = [""]
        self.selected_weights:List[float] = [1.0]
        self.selected_modifier = None
        super().__init__(name)


    def getValue(self) -> Tuple[str|None, List[Tuple[str, float]]]:
        return (self.basemodel, list(zip(self.selected_models, self.selected_weights)))
    

    def setValue(self, value:Tuple[str|None, List[Tuple[str, float]]]):
        self.basemodel = value[0]
        models = value[1]
        try:
            if models is not None:
                self.selected_models = [v[0] for v in models]
                self.selected_weights = [v[1] for v in models]
        except:
            self.selected_models = [""]
            self.selected_weights = [1.0]
        self.fireUpdate()


    @ui.refreshable
    def gui(self):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines not initialised")  
        with ui.column().classes('grow'):
            ui.label(f"Models")
            with ui.row().classes('w-full'):
                self.basemodel_dropdown = ui.select(options=self.basemodels, value=self.basemodel, label=f"Base Model", on_change=lambda e: self.updateModels()).bind_value(self, 'basemodel').classes('grow')  
                ui.button(icon="settings", on_click=lambda e: self.modelSettings()).classes('align-middle').props('dense')
            for i in range(len(self.selected_models)):
                self.modelGui(i)
        with ui.row().classes('w-full'):
            ui.label().classes('w-8')
            ui.button(icon="add", on_click = lambda e: self.addInput(0)).props('dense').classes('align-middle')


    def modelGui(self, i):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines not initialised")  
        models = DiffusersPipelines.pipelines.presets.getModelsByTypeAndBase(self.modeltype, self.basemodel)
        selected_model = self.selected_models[i]
        if selected_model not in models:
            selected_model = None
        with ui.row().classes('w-full'):
            ui.select(options=sorted(list(models)), with_input=True, value=selected_model, label="Model", on_change=lambda e: self.updateModel(i, e.value)).classes('grow') 
            ui.number(value=self.selected_weights[i], label="Weight", on_change=lambda e: self.updateWeight(i, e.value)).classes('small-number')
            ui.button(icon="remove", on_click = lambda e: self.removeInput(i)).props('dense').classes('align-middle')


    def addInput(self, i):
        self.selected_models.append("")
        self.selected_weights.append(1.0)
        self.gui.refresh()


    def removeInput(self, i):
        self.selected_models.pop(i)
        self.selected_weights.pop(i)
        self.gui.refresh()


    def updateModel(self, i, value):
        self.selected_models[i] = value
        self.fireUpdate()


    def updateWeight(self, i, value):
        self.selected_weights[i] = value


    def updateModels(self):
        self.fireUpdate()
        self.gui.refresh()


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


    def process(self) -> ModelsType:
        if(len(self.selected_models) == 0):
            raise Exception("Model not selected")
        modelparams:List[ModelParameters] = []
        for i, selected_model in enumerate(self.selected_models):
            weight = self.selected_weights[i]
            if selected_model[0] != "":
                modelparams.append(ModelParameters(selected_model, weight))
        return modelparams