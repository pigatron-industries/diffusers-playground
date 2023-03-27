from ..batch import BatchRunner, RandomNumberArgument
from ..inference import DiffusersPipeline
import ipywidgets as widgets
from IPython.display import display, clear_output

class BatchNotebookInterface:
    def __init__(self, pipelines:DiffusersPipeline, output_dir):
        self.pipelines = pipelines
        self.output_dir = output_dir

        self.type_dropdown = self.dropdown(label="Type:", options=["Text to image", "Image to image", "Control Net"], value="Text to image")
        self.model_dropdown = self.dropdown(label="Model:", options=pipelines.presetsImage.models.keys(), value=None)
        self.initimagetype_dropdown = self.dropdown(label="Init Image:", options=["Image Single", "Image Folder", "Generated"], value="Image Single")
        self.prompt_text = self.textarea(label="Prompt:", value="")
        self.shuffle_checkbox = self.checkbox(label="Shuffle", value=False)
        self.negprompt_text = self.textarea(label="Neg Prompt:", value="")
        self.width_slider = self.intSlider(label='Width:', value=512, min=256, max=1024, step=64)
        self.height_slider = self.intSlider(label='Height:', value=512, min=256, max=1024, step=64)
        self.scale_slider = self.floatSlider(label='Guidance:', value=9, min=1, max=20, step=0.1)
        self.steps_slider = self.intSlider(label='Steps:', value=40, min=5, max=100, step=5)
        self.strength_slider = self.floatSlider(label='Strength:', value=0.5, min=0, max=1, step=0.01)
        self.seed_text = self.intText(label='Seed:', value=None)
        self.batchsize_slider = self.intSlider(label='Batch:', value=10, min=1, max=100, step=1)

        # TODO sampler, tiling, apply RandomPromptProcessor with modifier dict

        self.setWidgetVisibility()
        display(self.type_dropdown, 
                self.model_dropdown, 
                self.initimagetype_dropdown, 
                self.prompt_text, 
                self.shuffle_checkbox,
                self.negprompt_text, 
                self.width_slider, 
                self.height_slider, 
                self.scale_slider,
                self.steps_slider,
                self.strength_slider,
                self.seed_text,
                self.batchsize_slider
        )


    def intText(self, label, value):
        inttext = widgets.IntText(
            value=value,
            description=label,
            disabled=False
        )
        inttext.observe(self.onChange)
        return inttext

    def intSlider(self, label, value, min, max, step):
        slider = widgets.IntSlider(
            value=value,
            min=min,
            max=max,
            step=step,
            description=label,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout={'width': '500px'}
        )
        slider.observe(self.onChange)
        return slider

    def floatSlider(self, label, value, min, max, step):
        slider = widgets.FloatSlider(
            value=value,
            min=min,
            max=max,
            step=step,
            description=label,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            layout={'width': '500px'}
        )
        slider.observe(self.onChange)
        return slider

    def dropdown(self, label, options, value):
        dropdown = widgets.Dropdown(
            options=options,
            description=label,
            value=value,
            layout={'width': '500px'}
        )
        dropdown.observe(self.onChange)
        return dropdown
    
    def textarea(self, label, value):
        textarea = widgets.Textarea(
            value=value,
            description=label,
            layout={'width': '500px'}
        )
        textarea.observe(self.onChange)
        return textarea
    
    def checkbox(self, label, value):
        checkbox = widgets.Checkbox(
            value=value,
            description=label,
            disabled=False,
            indent=True
        )
        checkbox.observe(self.onChange)
        return checkbox


    def setWidgetVisibility(self):
        if(self.type_dropdown.value == "Image to image"):
            self.initimagetype_dropdown.layout.visibility = 'visible'
            self.strength_slider.layout.visibility = 'visible'
            self.steps_slider.layout.visibility = 'hidden'
        else:
            self.initimagetype_dropdown.layout.visibility = 'hidden'
            self.strength_slider.layout.visibility = 'hidden'
            self.steps_slider.layout.visibility = 'visible'

    
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
        params['scale'] = self.scale_slider.value
        params['batch'] = self.batchsize_slider.value

        if(self.type_dropdown.value == "Image to image"):
            params['strength'] = self.strength_slider.value
        else:
            params['steps'] = self.steps_slider.value

        if(self.seed_text.value > 0):
            params['seed'] = self.seed_text.value
        else:
            params['seed'] = RandomNumberArgument(0, 4294967295)

        return params
    
    def run(self):
        params = self.getParams()
        if(self.type_dropdown.value == "Text to image"):
            batch = BatchRunner(self.pipelines.textToImage, params, params['batch'], self.output_dir)    
        elif(self.type_dropdown.value == "Image to image"):
            batch = BatchRunner(self.pipelines.imageToImage, params, params['batch'], self.output_dir)
        batch.run()