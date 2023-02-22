from ..batch import BatchRunner
from ..inference import DiffusersPipeline
import ipywidgets as widgets
from IPython.display import display

class BatchNotebookInterface:
    def __init__(self, pipelines:DiffusersPipeline):
        self.pipelines = pipelines

        self.type_dropdown = widgets.Dropdown(
            options = ["Text to image", "Image to image"],
            value="Text to image",
            description = "Type:",
            layout = {'width': '500px'}
        )
        self.model_dropdown = widgets.Dropdown(
            options=pipelines.presetsImage.models.keys(),
            description='Model:',
            layout={'width': '500px'}
        )
        self.initimagetype_dropdown = widgets.Dropdown(
            options=["Image Single", "Image Folder", "Generated"],
            description='Init Image:',
            layout={'width': '500px'}
        )

        self.type_dropdown.observe(self.onChange)
        self.model_dropdown.observe(self.onChange)
        self.initimagetype_dropdown.observe(self.onChange)

        self.setWidgetVisibility()
        display(self.type_dropdown, self.model_dropdown, self.initimagetype_dropdown)


    def setWidgetVisibility(self):
        if(self.type_dropdown.value == "Image to image"):
            self.initimagetype_dropdown.layout.visibility = 'visible'
        else:
            self.initimagetype_dropdown.layout.visibility = 'hidden'

    
    def onChange(self, change):
        if (change['type'] == 'change' and change['name'] == 'value'):
            self.setWidgetVisibility()


    def getParams(self):
        params = {}
        params['type'] = self.type_dropdown.value
        params['model'] = self.model_dropdown.value
        return params