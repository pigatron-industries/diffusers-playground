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
        self.prompt_text = widgets.Textarea(
            value='',
            placeholder='Prompt',
            description='Prompt:',
            layout={'width': '500px'}
        )
        self.negprompt_text = widgets.Textarea(
            value='',
            placeholder='Neg Prompt',
            description='Neg Prompt:',
            layout={'width': '500px'}
        )
        self.width_slider = widgets.IntSlider(
            value=512,
            min=256,
            max=1024,
            step=64,
            description='Width:',
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout={'width': '500px'}
        )
        self.height_slider = widgets.IntSlider(
            value=512,
            min=256,
            max=1024,
            step=64,
            description='Height:',
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout={'width': '500px'}
        )



        self.type_dropdown.observe(self.onChange)
        self.model_dropdown.observe(self.onChange)
        self.initimagetype_dropdown.observe(self.onChange)
        self.prompt_text.observe(self.onChange)
        self.negprompt_text.observe(self.onChange)
        self.width_slider.observe(self.onChange)
        self.height_slider.observe(self.onChange)

        self.setWidgetVisibility()
        display(self.type_dropdown, self.model_dropdown, self.initimagetype_dropdown, self.prompt_text, self.negprompt_text, self.width_slider, self.height_slider)


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
        params['prompt'] = self.prompt_text.value
        params['negprompt'] = self.negprompt_text.value
        params['width'] = self.width_slider.value
        params['height'] = self.width_slider.value
        return params