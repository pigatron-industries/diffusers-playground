from . import BatchRunner, RandomNumberArgument, RandomPromptProcessor, RandomImage
from ..inference import DiffusersPipelines, LORAUse
from ..FileUtils import getLeafFolders
from ..processing import *
import ipywidgets as widgets
import pickle 
import os
from typing import List, Dict
from functools import partial
from IPython.display import HTML, display, clear_output

INTERFACE_WIDTH = '900px'

DEFAULT_PREPROCESSORS = {
    'canny edge detection': CannyEdgeProcessor,
    'holistically nested edge detection': EdgeDetectionProcessor,
    'straight line detection': StraightLineDetectionProcessor,
    'pose detection': PoseDetectionProcessor,
    'depth estimation': DepthEstimationProcessor,
    'normal estimation': NormalEstimationProcessor,
    'segmentation': SegmentationProcessor,
    'monochrome': partial(SaturationProcessor, saturation=0),
    'blur': partial(GaussianBlurProcessor, radius=2),
    'noise': partial(GaussianNoiseProcessor, sigma=10),
}

class BatchNotebookInterface:
    def __init__(self, pipelines:DiffusersPipelines, output_dir:str, modifier_dict=None, save_file:str='batch_params.pkl', 
                 generation_pipelines:Dict[str, ImageProcessorPipeline]={}, 
                 preprocessing_pipelines:Dict[str, ImageProcessor]=DEFAULT_PREPROCESSORS, 
                 input_dirs:List[str]=[]):
        self.pipelines = pipelines
        self.output_dir = output_dir
        self.input_dirs = input_dirs
        self.modifier_dict = modifier_dict
        self.save_file = save_file
        self.generation_pipelines = generation_pipelines
        self.preprocessing_pipelines = preprocessing_pipelines

        self.type_dropdown = self.dropdown(label="Type:", options=["Text to image", "Image to image", "Control Net"], value="Text to image")

        # Control Net
        # TODO filter list to only show control nets for selected base model
        self.controlmodel_dropdown = self.dropdown(label="Control Model:", options=list(pipelines.presetsControl.models.keys()), value=None)

        #  Init images
        self.generationpipeline_dropdown = self.dropdown(label="Generation:", options=list(generation_pipelines.keys()), value=None)
        self.generationinput_dropdown = self.dropdown(label="Input:", options=input_dirs, value=None)
        self.preprocessing_dropdown = self.dropdown(label="Preprocessor:", options=[None]+list(self.preprocessing_pipelines.keys()), value=None)

        self.model_dropdown = self.dropdown(label="Model:", options=list(pipelines.presetsImage.models.keys()), value=None)
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

        html = HTML('''<style>
                        .widget-label { min-width: 20ex !important; }
                    </style>''')

        display(html, 
                self.type_dropdown, 
                self.controlmodel_dropdown,
                self.generationpipeline_dropdown,
                self.generationinput_dropdown,
                self.preprocessing_dropdown,
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
            self.generationpipeline_dropdown.layout.visibility = 'hidden'
            self.generationinput_dropdown.layout.visibility = 'hidden'
            self.preprocessing_dropdown.layout.visibility = 'hidden'
        else:
            self.generationpipeline_dropdown.layout.visibility = 'visible'
            self.preprocessing_dropdown.layout.visibility = 'visible'
            if(self.generationpipeline_dropdown.value is not None and self.generation_pipelines[self.generationpipeline_dropdown.value].requireInputImage()):
                self.generationinput_dropdown.layout.visibility = 'visible'
            else:
                self.generationinput_dropdown.layout.visibility = 'hidden'

        if(self.type_dropdown.value == "Image to image"):
            self.strength_slider.layout.visibility = 'visible'
            self.steps_slider.layout.visibility = 'hidden'
        else:
            self.strength_slider.layout.visibility = 'hidden'
            self.steps_slider.layout.visibility = 'visible'

        if(self.type_dropdown.value == "Control Net"):
            self.controlmodel_dropdown.layout.visibility = 'visible'
        else:
            self.controlmodel_dropdown.layout.visibility = 'hidden'

    
    def onChange(self, change):
        if (change['type'] == 'change' and change['name'] == 'value'):
            self.updateWidgets()


    def getParams(self):
        params = {}
        params['type'] = self.type_dropdown.value
        params['model'] = self.model_dropdown.value
        params['init_prompt'] = self.prompt_text.value
        params['shuffle'] = self.shuffle_checkbox.value
        params['prompt'] = RandomPromptProcessor(self.modifier_dict, self.prompt_text.value, shuffle=self.shuffle_checkbox.value)
        params['negprompt'] = self.negprompt_text.value
        params['width'] = self.width_slider.value
        params['height'] = self.height_slider.value
        params['scale'] = self.scale_slider.value
        params['scheduler'] = self.scheduler_dropdown.value
        params['batch'] = self.batchsize_slider.value

        if(self.lora_dropdown.value is not None):
            params['lora'] = self.lora_dropdown.value
            params['lora_weight'] = self.loraweight_text.value

        if(self.type_dropdown.value != "Text to image"):
            params['generationpipeline'] = self.generationpipeline_dropdown.value
            self.preprocessing_dropdown.layout.visibility = 'visible'
            params['initimage'] = self.generation_pipelines[self.generationpipeline_dropdown.value]
            if(self.preprocessing_dropdown.value is not None):
                preprocessor = self.preprocessing_pipelines[self.preprocessing_dropdown.value]
                params['preprocessor'] = self.preprocessing_dropdown.value
                params['initimage'].addTask(preprocessor())
            if(self.generation_pipelines[self.generationpipeline_dropdown.value].requireInputImage()):
                params['generationinput'] = self.generationinput_dropdown.value
                params['initimage'].setInputImage(RandomImage.fromDirectory(self.generationinput_dropdown.value))

        if(self.type_dropdown.value == "Image to image"):
            params['strength'] = self.strength_slider.value
        else:
            params['steps'] = self.steps_slider.value

        if(self.type_dropdown.value == "Control Net"):
            params['controlmodel'] = self.controlmodel_dropdown.value

        if(self.seed_text.value > 0):
            params['seed'] = self.seed_text.value
        else:
            params['seed'] = RandomNumberArgument(0, 4294967295)

        return params
    

    def setParams(self, params):        
        self.type_dropdown.value = params.get('type', 'Text to image')
        self.controlmodel_dropdown.value = params.get('controlmodel', None)
        self.generationpipeline_dropdown.value = params.get('generationpipeline', None)
        self.generationinput_dropdown.value = params.get('generationinput', None)
        self.preprocessing_dropdown.value = params.get('preprocessor', None)
        self.model_dropdown.value = params.get('model', None)
        self.lora_dropdown.value = params.get('lora', None)
        self.loraweight_text.value = params.get('lora_weight', 1)
        self.prompt_text.value = params.get('init_prompt', '')
        self.negprompt_text.value = params.get('negprompt', '')
        self.width_slider.value = params.get('width', 512)
        self.height_slider.value = params.get('height', 512)
        self.scale_slider.value = params.get('scale', 9.0)
        self.steps_slider.value = params.get('steps', 40)
        self.strength_slider.value = params.get('strength', 0.5)
        self.scheduler_dropdown.value = params.get('scheduler', 'EulerDiscreteScheduler')
        self.batchsize_slider.value = params.get('batch', 10)
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
        elif(self.type_dropdown.value == "Control Net"):
            batch = BatchRunner(self.pipelines.controlNet, params, params['batch'], self.output_dir)
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
            readout_format='.2f',
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