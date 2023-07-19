from ..batch import BatchRunner
from ..batch.argument import RandomNumberArgument, RandomImageArgument, RandomPromptProcessor
from ..inference import DiffusersPipelines, LORAUse
from ..processing import *
from ..processing.ProcessingPipeline import ImageProcessorPipeline
from ..processing.processors.FilterProcessors import *
from ..processing.processors.TransformerProcessors import *
import ipywidgets as widgets
import pickle 
import os
import copy
import glob
import functools
from collections import OrderedDict
from typing import List, Dict, Callable
from functools import partial
from IPython.display import display, clear_output
from PIL import Image

INTERFACE_WIDTH = '900px'

DEFAULT_PREPROCESSORS = {
    'canny edge detection': CannyEdgeProcessor,
    'HED edge detection': HEDEdgeDetectionProcessor,
    'PIDI edge detection': PIDIEdgeDetectionProcessor,
    'MLSD straight line detection': MLSDStraightLineDetectionProcessor,
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
RANDOM_IMAGE = "Random"


class InitImageWidgets:
    def __init__(self, interface, firstImage = False):
        self.interface = interface
        self.firstImage = firstImage
        controlnet_models = list(interface.pipelines.presets.getModelsByType("controlnet").keys())
        if (firstImage):
            controlnet_models = [INIT_IMAGE] + controlnet_models
        generation_pipeline_options = list(interface.generation_pipelines.keys())
        inputdirs_options = interface.input_dirs
        if (not firstImage):
            inputdirs_options = [PREV_IMAGE] + inputdirs_options
        
        self.model_dropdown = interface.dropdown(label="Control Model:", options=controlnet_models, value=INIT_IMAGE if firstImage else None)
        self.generation_dropdown = interface.dropdown(label="Generation:", options=generation_pipeline_options, value=None)
        self.input_source_dropdown = interface.dropdown(label="Input Source:", options=inputdirs_options, value=None)
        self.input_select_dropdown = interface.dropdown(label="Input Select:", options=[], value=None)
        self.preprocessor_dropdown = interface.dropdown(label="Preprocessor:", options=[None]+list(interface.preprocessing_pipelines.keys()), value=None)

    def display(self):
        display(self.model_dropdown,
                self.generation_dropdown,
                self.input_source_dropdown,
                self.input_select_dropdown,
                self.preprocessor_dropdown,
                widgets.HTML("<span>&nbsp;</span>"))
        
    def updateWidgets(self):
        if (self.input_source_dropdown.value is not None and self.input_source_dropdown.value != PREV_IMAGE):
            filepaths = glob.glob(f"{self.input_source_dropdown.value}/*.png") + glob.glob(f"{self.input_source_dropdown.value}/*.jpg")
            self.input_select_dropdown.options = [RANDOM_IMAGE] + [os.path.basename(x) for x in filepaths]
        else:
            self.input_select_dropdown.options = []
        
    def hide(self):
        self.model_dropdown.layout.display = 'none'
        self.generation_dropdown.layout.display = 'none'
        self.input_source_dropdown.layout.display = 'none'
        self.input_select_dropdown.layout.display = 'none'
        self.preprocessor_dropdown.layout.display = 'none'

    def show(self):
        self.model_dropdown.layout.display = 'block'
        self.generation_dropdown.layout.display = 'block'
        self.input_source_dropdown.layout.display = 'block'
        self.input_select_dropdown.layout.display = 'block'
        self.preprocessor_dropdown.layout.display = 'block'


    def getInitImage(self):
        if(self.input_select_dropdown.value == RANDOM_IMAGE):
            return RandomImageArgument.fromDirectory(self.input_source_dropdown.value)
        else:
            return Image.open(self.input_source_dropdown.value + "/" + self.input_select_dropdown.value)
        

    def createGenerationPipeline(self, prevImageFunc:Callable[[], (Image.Image|None)]|Image.Image|None = None, feedbackImage = False) -> ImageProcessorPipeline:
        pipeline = self.interface.generation_pipelines[self.generation_dropdown.value]
        pipeline = copy.deepcopy(pipeline)
        if(self.preprocessor_dropdown.value is not None):
            preprocessor = self.interface.preprocessing_pipelines[self.preprocessor_dropdown.value]
            pipeline.addTask(preprocessor())
        if(pipeline.hasPlaceholder("image")):
            if(self.input_source_dropdown.value != PREV_IMAGE and not feedbackImage):
                pipeline.setPlaceholder("image", self.getInitImage())
            elif (prevImageFunc is not None):
                pipeline.setPlaceholder("image", prevImageFunc)
        if(pipeline.hasPlaceholder("size")):
            pipeline.setPlaceholder("size", (self.interface.width_slider.value, self.interface.height_slider.value))
        return pipeline


class LoraWidgets:
    def __init__(self, interface):
        self.interface = interface
        self.lora_dropdown = interface.dropdown(label="LORA:", options=[""], value=None)
        self.loraweight_text = interface.floatText(label="LORA weight:", value=1)

    def display(self):
        display(self.lora_dropdown,
                self.loraweight_text)
        
    def hide(self):
        self.lora_dropdown.layout.display = 'none'
        self.loraweight_text.layout.display = 'none'

    def show(self):
        self.lora_dropdown.layout.display = 'flex'
        self.loraweight_text.layout.display = 'flex'


class BatchNotebookInterface:
    def __init__(self, pipelines:DiffusersPipelines, output_dir:str, modifier_dict=None, save_file:str='batch_params.pkl', 
                 generation_pipelines:Dict[str, ImageProcessorPipeline]={}, 
                 preprocessing_pipelines:Dict[str, ImageProcessor]=DEFAULT_PREPROCESSORS, 
                 input_dirs:List[str]=[]):
        self.images = []
        self.pipelines = pipelines
        self.output_dir = output_dir
        self.input_dirs = input_dirs
        self.modifier_dict = modifier_dict
        self.save_file = save_file
        self.generation_pipelines = generation_pipelines
        self.preprocessing_pipelines = preprocessing_pipelines

        #  Init images
        self.initimages_num = self.intSlider(label='Input Images:', value=0, min=0, max=4, step=1)
        self.initimage_widgets:List[InitImageWidgets] = []
        self.initimage_widgets.append(InitImageWidgets(self, firstImage=True))
        self.initimage_widgets.append(InitImageWidgets(self))
        self.initimage_widgets.append(InitImageWidgets(self))
        self.initimage_widgets.append(InitImageWidgets(self))

        # Model options
        self.model_dropdown = self.dropdown(label="Model:", options=list(pipelines.presets.getModelsByType("txt2img").keys()), value=None)
        self.mergemodel_dropdown = self.dropdown(label="Model Merge:", options=[None] + list(pipelines.presets.getModelsByType("txt2img").keys()), value=None)
        self.mergeweight_slider = self.floatSlider(label='Merge Weight:', value=0.5, min=0, max=1, step=0.01)
        self.lora_num = self.intSlider(label='LORAs:', value=0, min=0, max=4, step=1)
        self.lora_widgets = []
        self.lora_widgets.append(LoraWidgets(self))
        self.lora_widgets.append(LoraWidgets(self))
        self.lora_widgets.append(LoraWidgets(self))
        self.lora_widgets.append(LoraWidgets(self))

        # Generation options
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

        self.run_button = widgets.Button(description="Run")
        self.clear_button = widgets.Button(description="Clear")
        self.output = widgets.Output()

        self.run_button.on_click(self._runClick)
        self.clear_button.on_click(self._clearClick)

        html = widgets.HTML('''<style>
                        .widget-label { min-width: 20ex !important; }
                    </style>''')

        display(html,
                self.initimages_num)
        for initimage_w in self.initimage_widgets:
            initimage_w.display()
        display(widgets.HTML("<span>&nbsp;</span>"),
                self.model_dropdown, 
                self.mergemodel_dropdown,
                self.mergeweight_slider,
                self.lora_num)
        for lora_w in self.lora_widgets:
            lora_w.display()
        display(widgets.HTML("<span>&nbsp;</span>"),
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
                self.batchsize_slider,
                widgets.HTML("<span>&nbsp;</span>"),
                self.run_button,
                self.clear_button,
                self.output
        )
        self.loadParams()


    def updateWidgets(self):
        for i, initimage_w in enumerate(self.initimage_widgets):
            if(i < self.initimages_num.value):
                initimage_w.show()
            else:
                initimage_w.hide()

        if(self.mergemodel_dropdown.value is not None):
            self.mergeweight_slider.layout.display = 'flex'
        else:
            self.mergeweight_slider.layout.display = 'none'

        for i, lora_w in enumerate(self.lora_widgets):
            if(i < self.lora_num.value):
                lora_w.show()
                if(self.model_dropdown.value is not None):
                    lora_w.lora_dropdown.options = [None] + self.pipelines.getLORAList(self.model_dropdown.value)
            else:
                lora_w.hide()

        if(self.initimages_num.value > 0 and self.initimage_widgets[0].model_dropdown.value == INIT_IMAGE):
            self.strength_slider.layout.display = 'flex'
            self.steps_slider.layout.display = 'none'
        else:
            self.strength_slider.layout.display = 'none'
            self.steps_slider.layout.display = 'flex'

        for initimage_w in self.initimage_widgets:
            initimage_w.updateWidgets()

    
    def onChange(self, change):
        if (change['type'] == 'change' and change['name'] == 'value'):
            self.updateWidgets()


    def getParams(self, initimage:Image.Image|None = None):
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
        params['initimages_num'] = self.initimages_num.value

        # image feedback handling
        prevImageFunc = None
        feebdackImage = False
        if(initimage is not None and params['initimages_num'] > 0):
            prevImageFunc = initimage
            feebdackImage = True
        else:
            pass #TODO add init image params

        for i, initimage_w in enumerate(self.initimage_widgets):
            if(i >= params['initimages_num']):
                break
            params[f'initimage{i}_model'] = initimage_w.model_dropdown.value
            params[f'initimage{i}_generation'] = initimage_w.generation_dropdown.value
            params[f'initimage{i}_input_source'] = initimage_w.input_source_dropdown.value
            params[f'initimage{i}_input_select'] = initimage_w.input_select_dropdown.value
            params[f'initimage{i}_preprocessor'] = initimage_w.preprocessor_dropdown.value
            pipeline = initimage_w.createGenerationPipeline(prevImageFunc, feebdackImage)
            prevImageFunc = pipeline.getLastOutput

            if(initimage_w.model_dropdown.value == INIT_IMAGE):
                params['initimage'] = pipeline
            else:
                if('controlimage' not in params):
                    params['controlimage'] = []
                if('controlmodel' not in params):
                    params['controlmodel'] = []
                params['controlimage'].append(pipeline)
                params['controlmodel'].append(initimage_w.model_dropdown.value)

        params['lora_num'] = self.lora_num.value
        for i, lora_w in enumerate(self.lora_widgets):
            if(i >= self.lora_num.value):
                break
            params[f'lora{i}_lora'] = lora_w.lora_dropdown.value
            params[f'lora{i}_loraweight'] = lora_w.loraweight_text.value

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
                initimage_w.input_source_dropdown.value = params.get(f'initimage{i}_input_source', None)
                initimage_w.input_select_dropdown.value = params.get(f'initimage{i}_input_select', None)
                initimage_w.preprocessor_dropdown.value = params.get(f'initimage{i}_preprocessor', None)

            model = params.get('model', None)
            if(isinstance(model, list)):
                self.model_dropdown.value = model[0]
                self.mergemodel_dropdown.value = model[1]
            else:
                self.model_dropdown.value = model
            self.mergeweight_slider.value = params.get('model_weight', 1)

            self.lora_num.value = params.get('lora_num', 0)
            for i, lora_w in enumerate(self.lora_widgets):
                lora_w.lora_dropdown.value = params.get(f'lora{i}_lora', None)
                lora_w.loraweight_text.value = params.get(f'lora{i}_loraweight', 1)

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
        filtered_params = { key: value for key, value in params.items() if key not in ['initimage', 'controlimage'] }
        with open(self.save_file, 'wb') as f:
            pickle.dump(filtered_params, f)
        return params


    def loadParams(self):
        if(os.path.isfile(self.save_file)):
            with open(self.save_file, 'rb') as f:
                params = pickle.load(f)
                self.setParams(params)
        self.updateWidgets()


    def _callback(self, image, output):
        with output:
            self.images.append(image)
            index = len(self.images)-1
            refineBtn = widgets.Button(description="Refine")
            refineBtn.on_click(functools.partial(self._refineClick, index))
            display(refineBtn)


    def _runClick(self, b):
        with self.output:
            self.run()


    def _clearClick(self, b):
        self.output.clear_output()
        self.images = []


    def _refineClick(self, index, b):
        with self.output:
            self.refine(self.images[index])


    def initPipeline(self, params):
        if(self.initimages_num.value == 1 and self.initimage_widgets[0].model_dropdown.value == INIT_IMAGE):
            pipelineFunc = self.pipelines.imageToImage
        elif(self.initimages_num.value > 1 and self.initimage_widgets[0].model_dropdown.value == INIT_IMAGE):
            pipelineFunc = self.pipelines.imageToImageControlNet
        elif(self.initimages_num.value == 0):
            pipelineFunc = self.pipelines.textToImage
        else:
            pipelineFunc = self.pipelines.textToImageControlNet
        self.batch = BatchRunner(pipelineFunc, self.output_dir, self._callback)

        loras = []
        for i, lora_w in enumerate(self.lora_widgets):
            if(i >= self.lora_num.value):
                break
            loras.append(LORAUse(params[f'lora{i}_lora'], params[f'lora{i}_loraweight']))
        self.pipelines.useLORAs(loras)


    def run(self):
        params = self.saveParams()
        self.initPipeline(params)
        self.batch.appendBatchArguments(params, params['batch'])
        self.batch.run()


    def refine(self, image):
        params = self.getParams(image)
        self.initPipeline(params)
        self.batch.appendBatchArguments(params, params['batch'])
        self.batch.run()



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