from . import BatchRunner, RandomNumberArgument, RandomPromptProcessor, RandomImage
from ..inference import DiffusersPipeline, LORAUse
from ..processing import simpleTransform
import ipywidgets as widgets
import pickle 
import os
from IPython.display import display, clear_output

INTERFACE_WIDTH = '900px'

class BatchNotebookInterface:
    def __init__(self, pipelines:DiffusersPipeline, output_dir, modifier_dict=None, save_file='batch_params.pkl', processing_pipelines={}):
        self.pipelines:DiffusersPipeline = pipelines
        self.output_dir = output_dir
        self.modifier_dict = modifier_dict
        self.save_file = save_file
        self.processing_pipelines = processing_pipelines

        self.type_dropdown = self.dropdown(label="Type:", options=["Text to image", "Image to image", "Control Net"], value="Text to image")

        #  Init images
        self.processingpipeline_dropdown = self.dropdown(label="Processing:", options=list(processing_pipelines.keys()), value=None)

        self.model_dropdown = self.dropdown(label="Model:", options=pipelines.presetsImage.models.keys(), value=None)
        self.lora_dropdown = self.dropdown(label="LORA:", options=[""], value=None)
        self.loraweight_text = self.floatText(label="LORA weight:", value=1)
        self.prompt_text = self.textarea(label="Prompt:", value="")
        self.shuffle_checkbox = self.checkbox(label="Shuffle", value=False)
        self.negprompt_text = self.textarea(label="Neg Prompt:", value="")
        self.width_slider = self.intSlider(label='Width:', value=512, min=256, max=1024, step=64)
        self.height_slider = self.intSlider(label='Height:', value=768, min=256, max=1024, step=64)
        self.scale_slider = self.floatSlider(label='Guidance:', value=9, min=1, max=20, step=0.1)
        self.steps_slider = self.intSlider(label='Steps:', value=40, min=5, max=100, step=5)
        self.strength_slider = self.floatSlider(label='Strength:', value=0.5, min=0, max=1, step=0.01)
        self.scheduler_dropdown = self.dropdown(label="Sampler:", options=['DDIMScheduler', 'DPMSolverMultistepScheduler', 
                                                                           'EulerAncestralDiscreteScheduler', 'EulerDiscreteScheduler',
                                                                           'LMSDiscreteScheduler', 'UniPCMultistepScheduler'], value="EulerDiscreteScheduler")
        self.seed_text = self.intText(label='Seed:', value=None)
        self.batchsize_slider = self.intSlider(label='Batch:', value=10, min=1, max=100, step=1)

        display(self.type_dropdown, 
                self.processingpipeline_dropdown,
                self.model_dropdown, 
                self.lora_dropdown,
                self.loraweight_text,
                self.prompt_text, 
                self.shuffle_checkbox,
                self.negprompt_text, 
                self.width_slider, 
                self.height_slider, 
                self.scale_slider,
                self.steps_slider,
                self.strength_slider,
                self.scheduler_dropdown,
                self.seed_text,
                self.batchsize_slider
        )
        self.loadParams()


    def updateWidgets(self):
        if(self.model_dropdown.value is not None):
            self.lora_dropdown.options = [None] + self.pipelines.getLORAList(self.model_dropdown.value)

        if(self.lora_dropdown.value is not None):
            self.loraweight_text.layout.visibility = 'visible'
        else:
            self.loraweight_text.layout.visibility = 'hidden'

        if(self.type_dropdown.value == "Text to image"):
            self.processingpipeline_dropdown.layout.visibility = 'hidden'
            self.strength_slider.layout.visibility = 'hidden'
            self.steps_slider.layout.visibility = 'visible'
        else:
            self.processingpipeline_dropdown.layout.visibility = 'visible'
            self.strength_slider.layout.visibility = 'visible'
            self.steps_slider.layout.visibility = 'hidden'

    
    def onChange(self, change):
        if (change['type'] == 'change' and change['name'] == 'value'):
            self.updateWidgets()


    def getParams(self):
        params = {}
        params['type'] = self.type_dropdown.value
        params['model'] = self.model_dropdown.value
        params['lora'] = self.lora_dropdown.value
        params['lora_weight'] = self.loraweight_text.value
        params['init_prompt'] = self.prompt_text.value
        params['prompt'] = RandomPromptProcessor(self.modifier_dict, self.prompt_text.value, shuffle=self.shuffle_checkbox.value)
        params['negprompt'] = self.negprompt_text.value
        params['width'] = self.width_slider.value
        params['height'] = self.height_slider.value
        params['scale'] = self.scale_slider.value
        params['scheduler'] = self.scheduler_dropdown.value
        params['batch'] = self.batchsize_slider.value

        if(self.type_dropdown.value != "Text to image"):
            params['processingpipeline'] = self.processingpipeline_dropdown.value
            params['initimage'] = self.processing_pipelines[self.initimagefile_text.value]

        if(self.type_dropdown.value == "Image to image"):
            params['strength'] = self.strength_slider.value
        else:
            params['steps'] = self.steps_slider.value

        if(self.seed_text.value > 0):
            params['seed'] = self.seed_text.value
        else:
            params['seed'] = RandomNumberArgument(0, 4294967295)

        return params
    

    def setParams(self, params):        
        self.type_dropdown.value = params['type']
        self.processingpipeline_dropdown.value = params['processingpipeline']
        self.model_dropdown.value = params['model']
        self.lora_dropdown.value = params['lora']
        self.loraweight_text.value = params['lora_weight']
        self.prompt_text.value = params['init_prompt']
        self.negprompt_text.value = params['negprompt']
        self.width_slider.value = params['width']
        self.height_slider.value = params['height']
        self.scale_slider.value = params['scale']
        self.scheduler_dropdown.value = params['scheduler']
        self.batchsize_slider.value = params['batch']
        self.updateWidgets()


    def saveParams(self):
        params = self.getParams()
        with open(self.save_file, 'wb') as f:
            pickle.dump(params, f)
        return params


    def loadParams(self):
        if(os.path.isfile(self.save_file)):
            with open(self.save_file, 'rb') as f:
                params = pickle.load(f)
                self.setParams(params)
        else:
            self.updateWidgets()
    

    def run(self):
        params = self.saveParams()
        if(self.lora_dropdown.value is not None):
            self.pipelines.useLORAs([LORAUse(params['lora'], params['lora_weight'])])
        else:
            self.pipelines.useLORAs([])

        if(self.type_dropdown.value == "Text to image"):
            batch = BatchRunner(self.pipelines.textToImage, params, params['batch'], self.output_dir)    
        elif(self.type_dropdown.value == "Image to image"):
            batch = BatchRunner(self.pipelines.imageToImage, params, params['batch'], self.output_dir)
        batch.run()



    # ============== widget helpers =============

    def text(self, label, value):
        text = widgets.Text(
            value=value,
            description=label,
            disabled=False,
            layout={'width': INTERFACE_WIDTH}
        )
        text.observe(self.onChange)
        return text

    def intText(self, label, value):
        inttext = widgets.IntText(
            value=value,
            description=label,
            disabled=False
        )
        inttext.observe(self.onChange)
        return inttext
    
    def floatText(self, label, value):
        floattext = widgets.FloatText(
            value=value,
            description=label,
            disabled=False
        )
        floattext.observe(self.onChange)
        return floattext

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
            layout={'width': INTERFACE_WIDTH}
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
            layout={'width': INTERFACE_WIDTH}
        )
        slider.observe(self.onChange)
        return slider

    def dropdown(self, label, options, value):
        dropdown = widgets.Dropdown(
            options=options,
            description=label,
            value=value,
            layout={'width': INTERFACE_WIDTH}
        )
        dropdown.observe(self.onChange)
        return dropdown
    
    def textarea(self, label, value):
        textarea = widgets.Textarea(
            value=value,
            description=label,
            layout={'width': INTERFACE_WIDTH, 'height': '100px'}
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