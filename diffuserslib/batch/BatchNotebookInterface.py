from . import BatchRunner, RandomNumberArgument, RandomImageArgument, RandomPromptProcessor
from ..inference import DiffusersPipelines, LORAUse
from ..FileUtils import getLeafFolders
from ..processing import *
import ipywidgets as widgets
import pickle 
import os
import copy
from collections import OrderedDict
from typing import List, Dict, Optional
from functools import partial
from IPython.display import display, clear_output

INTERFACE_WIDTH = '900px'

DEFAULT_PREPROCESSORS = {
    'canny edge detection': CannyEdgeProcessor,
    'holistically nested edge detection': EdgeDetectionProcessor,
    'straight line detection': StraightLineDetectionProcessor,
    'pose detection': PoseDetectionProcessor,
    'depth estimation': DepthEstimationProcessor,
    'normal estimation': NormalEstimationProcessor,
    'segmentation': SegmentationProcessor,
    'content shuffle': ContentShuffleProcessor,
    'monochrome': partial(SaturationProcessor, saturation=0),
    'blur': partial(GaussianBlurProcessor, radius=2),
    'noise': partial(GaussianNoiseProcessor, sigma=10),
}

INIT_IMAGE = "Init Image"
PREV_IMAGE = "Previous Image"


class InitImageWidgets:
    def __init__(self, interface, includeInitImage = False):
        self.interface = interface
        controlnet_models = list(interface.pipelines.presets.getModelsByType("controlnet").keys())
        if (includeInitImage):
            controlnet_models = [INIT_IMAGE] + controlnet_models
        generation_pipeline_options = list(interface.generation_pipelines.keys())
        inputdirs_options = interface.input_dirs
        if (not includeInitImage):
            inputdirs_options = [PREV_IMAGE] + inputdirs_options
        
        self.model_dropdown = interface.dropdown(label="Control Model:", options=controlnet_models, value=INIT_IMAGE if includeInitImage else None)
        self.generation_dropdown = interface.dropdown(label="Generation:", options=generation_pipeline_options, value=None)
        self.input_dropdown = interface.dropdown(label="Input:", options=inputdirs_options, value=None)
        self.preprocessor_dropdown = interface.dropdown(label="Preprocessor:", options=[None]+list(interface.preprocessing_pipelines.keys()), value=None)

    def display(self):
        display(self.model_dropdown,
                self.generation_dropdown,
                self.input_dropdown,
                self.preprocessor_dropdown,
                widgets.HTML("<span>&nbsp;</span>"))
        
    def hide(self):
        self.model_dropdown.layout.display = 'none'
        self.generation_dropdown.layout.display = 'none'
        self.input_dropdown.layout.display = 'none'
        self.preprocessor_dropdown.layout.display = 'none'

    def show(self):
        self.model_dropdown.layout.display = 'block'
        self.generation_dropdown.layout.display = 'block'
        self.input_dropdown.layout.display = 'block'
        self.preprocessor_dropdown.layout.display = 'block'

    def createGenerationPipeline(self, prevPipeline:Optional[ImageProcessorPipeline] = None) -> ImageProcessorPipeline:
        pipeline = self.interface.generation_pipelines[self.generation_dropdown.value]
        pipeline = copy.deepcopy(pipeline)
        if(self.preprocessor_dropdown.value is not None):
            preprocessor = self.interface.preprocessing_pipelines[self.preprocessor_dropdown.value]
            pipeline.addTask(preprocessor())
        if(pipeline.hasPlaceholder("image")):
            if(self.input_dropdown.value != PREV_IMAGE):
                pipeline.setPlaceholder("image", RandomImageArgument.fromDirectory(self.input_dropdown.value))
            else:
                if (prevPipeline is not None):
                    pipeline.setPlaceholder("image", prevPipeline.getLastOutput)
        if(pipeline.hasPlaceholder("size")):
            pipeline.setPlaceholder("size", (self.interface.width_slider.value, self.interface.height_slider.value))
        return pipeline


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

        #  Init images
        self.initimages_num = self.intSlider(label='Input Images:', value=0, min=0, max=4, step=1)

        # use an array of widgets instead
        self.initimage_widgets = []
        self.initimage_widgets.append(InitImageWidgets(self, includeInitImage=True))
        self.initimage_widgets.append(InitImageWidgets(self))
        self.initimage_widgets.append(InitImageWidgets(self))
        self.initimage_widgets.append(InitImageWidgets(self))

        #  Config
        self.model_dropdown = self.dropdown(label="Model:", options=list(pipelines.presets.getModelsByType("txt2img").keys()), value=None)
        self.mergemodel_dropdown = self.dropdown(label="Model Merge:", options=[None] + list(pipelines.presets.getModelsByType("txt2img").keys()), value=None)
        self.mergeweight_slider = self.floatSlider(label='Merge Weight:', value=0.5, min=0, max=1, step=0.01)
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

        html = widgets.HTML('''<style>
                        .widget-label { min-width: 20ex !important; }
                    </style>''')

        display(html,
                self.initimages_num,
                widgets.HTML("<span>&nbsp;</span>"))
        for initimage_w in self.initimage_widgets:
            initimage_w.display()
        display(self.model_dropdown, 
                self.mergemodel_dropdown,
                self.mergeweight_slider,
                self.lora_dropdown,
                self.loraweight_text,
                widgets.HTML("<span>&nbsp;</span>"),
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
        for i, initimage_w in enumerate(self.initimage_widgets):
            if(i < self.initimages_num.value):
                initimage_w.show()
            else:
                initimage_w.hide()

        if(self.model_dropdown.value is not None):
            self.lora_dropdown.options = [None] + self.pipelines.getLORAList(self.model_dropdown.value)

        if(self.mergemodel_dropdown.value is not None):
            self.mergeweight_slider.layout.display = 'flex'
        else:
            self.mergeweight_slider.layout.display = 'none'

        if(self.lora_dropdown.value is not None):
            self.loraweight_text.layout.display = 'flex'
        else:
            self.loraweight_text.layout.display = 'none'

        if(self.initimages_num.value > 0 and self.initimage_widgets[0].model_dropdown.value == INIT_IMAGE):
            self.strength_slider.layout.display = 'flex'
            self.steps_slider.layout.display = 'none'
        else:
            self.strength_slider.layout.display = 'none'
            self.steps_slider.layout.display = 'flex'

    
    def onChange(self, change):
        if (change['type'] == 'change' and change['name'] == 'value'):
            self.updateWidgets()


    def getParams(self):
        params = OrderedDict()
        if(self.mergemodel_dropdown.value is None):
            params['model'] = self.model_dropdown.value
        else:
            params['model'] = [self.model_dropdown.value, self.mergemodel_dropdown.value]
        params['model_weight'] = self.mergeweight_slider.value
        params['init_prompt'] = self.prompt_text.value
        params['shuffle'] = self.shuffle_checkbox.value
        params['prompt'] = RandomPromptProcessor(self.modifier_dict, str(self.prompt_text.value), shuffle=bool(self.shuffle_checkbox.value))
        params['negprompt'] = self.negprompt_text.value
        params['width'] = self.width_slider.value
        params['height'] = self.height_slider.value
        params['scale'] = self.scale_slider.value
        params['scheduler'] = self.scheduler_dropdown.value
        params['batch'] = self.batchsize_slider.value

        if(self.lora_dropdown.value is not None):
            params['lora'] = self.lora_dropdown.value
            params['lora_weight'] = self.loraweight_text.value

        params['initimages_num'] = self.initimages_num.value

        prevPipeline = None
        for i, initimage_w in enumerate(self.initimage_widgets):
            if(i >= self.initimages_num.value):
                break
            params[f'initimage{i}_model'] = initimage_w.model_dropdown.value
            params[f'initimage{i}_generation'] = initimage_w.generation_dropdown.value
            params[f'initimage{i}_input'] = initimage_w.input_dropdown.value
            params[f'initimage{i}_preprocessor'] = initimage_w.preprocessor_dropdown.value
            pipeline = initimage_w.createGenerationPipeline(prevPipeline)
            prevPipeline = pipeline

            if(initimage_w.model_dropdown.value == INIT_IMAGE):
                params['initimage'] = pipeline
            else:
                if('controlimage' not in params):
                    params['controlimage'] = []
                if('controlmodel' not in params):
                    params['controlmodel'] = []
                params['controlimage'].append(pipeline)
                params['controlmodel'].append(initimage_w.model_dropdown.value)

        if(self.initimages_num.value > 0 and self.initimage_widgets[0].model_dropdown.value == INIT_IMAGE):
            params['strength'] = self.strength_slider.value
        else:
            params['steps'] = self.steps_slider.value

        if(self.seed_text.value > 0):
            params['seed'] = self.seed_text.value
        else:
            params['seed'] = RandomNumberArgument(0, 4294967295)

        return params
    

    def setParams(self, params):
        try:
            self.initimages_num.value = params.get('initimages_num', 0)
            for i, initimage_w in enumerate(self.initimage_widgets):
                initimage_w.model_dropdown.value = params.get(f'initimage{i}_model', None)
                initimage_w.generation_dropdown.value = params.get(f'initimage{i}_generation', None)
                initimage_w.input_dropdown.value = params.get(f'initimage{i}_input', None)
                initimage_w.preprocessor_dropdown.value = params.get(f'initimage{i}_preprocessor', None)

            model = params.get('model', None)
            if(isinstance(model, list)):
                self.model_dropdown.value = model[0]
                self.mergemodel_dropdown.value = model[1]
            else:
                self.model_dropdown.value = model
            self.mergeweight_slider.value = params.get('model_weight', 1)
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
        except Exception as e:
            print(e)
            print("Error loading params")
            pass
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

        if(self.initimages_num.value == 1 and self.initimage_widgets[0].model_dropdown.value == INIT_IMAGE):
            batch = BatchRunner(self.pipelines.imageToImage, params, params['batch'], self.output_dir)
        elif(self.initimages_num.value > 1 and self.initimage_widgets[0].model_dropdown.value == INIT_IMAGE):
            batch = BatchRunner(self.pipelines.imageToImageControlNet, params, params['batch'], self.output_dir)
        elif(self.initimages_num.value == 0):
            batch = BatchRunner(self.pipelines.textToImage, params, params['batch'], self.output_dir)
        else:
            batch = BatchRunner(self.pipelines.textToImageControlNet, params, params['batch'], self.output_dir)
        batch.run()



    # ============== widget helpers =============

    def text(self, label, value) -> widgets.Text:
        text = widgets.Text(
            value=value,
            description=label,
            disabled=False,
            layout={'width': INTERFACE_WIDTH}
        )
        text.observe(self.onChange)
        return text

    def intText(self, label, value) -> widgets.IntText:
        inttext = widgets.IntText(
            value=value,
            description=label,
            disabled=False
        )
        inttext.observe(self.onChange)
        return inttext
    
    def floatText(self, label, value) -> widgets.FloatText:
        floattext = widgets.FloatText(
            value=value,
            description=label,
            disabled=False
        )
        floattext.observe(self.onChange)
        return floattext

    def intSlider(self, label, value, min, max, step) -> widgets.IntSlider:
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

    def floatSlider(self, label, value, min, max, step) -> widgets.FloatSlider:
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

    def dropdown(self, label, options, value) -> widgets.Dropdown:
        dropdown = widgets.Dropdown(
            options=options,
            description=label,
            value=value,
            layout={'width': INTERFACE_WIDTH}
        )
        dropdown.observe(self.onChange)
        return dropdown
    
    def textarea(self, label, value) -> widgets.Textarea:
        textarea = widgets.Textarea(
            value=value,
            description=label,
            layout={'width': INTERFACE_WIDTH, 'height': '100px'}
        )
        textarea.observe(self.onChange)
        return textarea
    
    def checkbox(self, label, value) -> widgets.Checkbox:
        checkbox = widgets.Checkbox(
            value=value,
            description=label,
            disabled=False,
            indent=True
        )
        checkbox.observe(self.onChange)
        return checkbox