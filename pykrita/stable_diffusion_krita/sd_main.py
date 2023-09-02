from cmd import PROMPT
from multiprocessing import dummy
import urllib.request
import json
from krita import (Krita, Qt, QApplication, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout,  # type: ignore
                   QLineEdit, QPlainTextEdit, QPushButton, QMessageBox, QComboBox, QTabWidget, QWidget, qAlpha )
from PyQt5.Qt import QByteArray # type: ignore
from PyQt5.QtGui import QPixmap # type: ignore
import array
from copy import copy
from pathlib import Path

from .sd_common import *
from .sd_server import *
from .sd_config import *
from .sd_parameters import *

# Stable Diffusion Plugin fpr Krita
# (C) 2022, Nicolay Mausz
# MIT License
#
rPath= Path(Krita.instance().readSetting("","ResourceDirectory",""))
class ModifierData:    
    list=[]
    tags=[]

    def serialize(self):
        obj={"list":self.list,"tags":self.tags}
        return json.dumps(obj)
    
    def unserialize(self,str):
        obj=json.loads(str)
        self.list=obj["list"]
        self.tags=obj["tags"]

    def save(self):
        str=self.serialize()
        Krita.instance().writeSetting ("SDPlugin", "Modifiers", str)

    def load(self):
        str=Krita.instance().readSetting ("SDPlugin", "Modifiers",None)
        if (not str): return
        self.unserialize(str) 


def createSlider(dialog,layout,value,min,max,steps,divider):
    h_layout =  QHBoxLayout()
    slider = QSlider(Qt.Orientation.Horizontal, dialog)
    slider.setRange(min, max)

    slider.setSingleStep(steps)
    slider.setPageStep(steps)
    slider.setTickInterval
    label = QLabel(str(value), dialog)
    h_layout.addWidget(slider, stretch=9)
    h_layout.addWidget(label, stretch=1)
    if (divider!=1):
        slider.valueChanged.connect(lambda: slider.setValue(slider.value()//steps*steps) or label.setText( str(slider.value()/divider)))
    else:
        slider.valueChanged.connect(lambda: slider.setValue(slider.value()//steps*steps) or label.setText( str(slider.value())))
    slider.setValue(int(value))
    layout.addLayout(h_layout)
    return slider


class SDConfigDialog(QDialog):
    def __init__(self):
        super().__init__(None)
        self.config = SDConfig()
        self.setWindowTitle("Stable Diffusion Configuration")
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout = QVBoxLayout()
        link_label=QLabel('Webservice URL:')
        link_label.setOpenExternalLinks(True)
        self.layout.addWidget(link_label)
        self.url = QLineEdit()
        self.url.setText(self.config.url)    
        self.layout.addWidget(self.url)

        self.layout.addWidget(QLabel(''))
        self.layout.addWidget(QLabel('Select tool size'))
        h_layout_width=QHBoxLayout()

        h_layout_width.addWidget(QLabel('Width:'))
        self.width=createSlider(self, h_layout_width, self.config.width, 256, 2048, 64, 1)      
        h_layout_width.addWidget(self.width)
        self.layout.addLayout(h_layout_width)

        h_layout_height=QHBoxLayout()
        h_layout_height.addWidget(QLabel('Height:'))
        self.height=createSlider(self, h_layout_height, self.config.height, 256, 2048, 64, 1)
        h_layout_height.addWidget(self.height)
        self.layout.addLayout(h_layout_height)

        self.layout.addWidget(QLabel(''))
        self.layout.addWidget(QLabel(''))

        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        self.resize(500,200)

    def save(self):
        self.config.url=self.url.text()
        self.config.width=self.width.value()
        self.config.height=self.height.value()
        self.config.save()


class ModifierDialog(QDialog):
    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("Stable Diffusion Modifier list")
        self.resize(800,500)
        QBtn =  QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.rejected.connect(self.reject)
        grid_layout = QVBoxLayout()
        self.layout=QVBoxLayout()
        self.setLayout(self.layout)

        ModifierData.load(ModifierData)
        self.layout.addLayout(grid_layout)
        for i in range(0,len(ModifierData.list)):
            entry=ModifierData.list[i]
            mod_layout=QVBoxLayout()
            mod_layout.addWidget(QLabel(entry["name"]))
            btn_h_layout=QHBoxLayout()
            btnSelect = QPushButton("")
            btnSelect.setIcon( Krita.instance().icon('select'))        
            btn_h_layout.addWidget(btnSelect,stretch=1)
            btnDelete = QPushButton("")
            btnDelete.setIcon( Krita.instance().icon('deletelayer'))   
            btnSelect.clicked.connect(lambda ch, num=i: self.selectModifier(num))
            btnDelete.clicked.connect(lambda ch, num=i: self.deleteModifier(num))

            btn_h_layout.addWidget(btnDelete,stretch=1)
            btn_h_layout.addWidget(QLabel(""),stretch=5)

            mod_layout.addLayout(btn_h_layout)
            grid_layout.addLayout(mod_layout)

        self.layout.addWidget(QLabel("Name"))
        self.name = QLineEdit()
#        self.name.setText(selected_mod["name"])       
        self.layout.addWidget(self.name) 
   
        self.layout.addWidget(QLabel("Modifiers"))
        self.modifiers=QPlainTextEdit()
        self.modifiers.setPlainText(SDConfig.params.modifiers)      
        self.layout.addWidget(self.modifiers)     

        self.layout.addWidget(QLabel("Example Prompt"))
        self.example_prompt = QLineEdit()
        self.example_prompt.setText(SDConfig.params.prompt)
        self.layout.addWidget(self.example_prompt) 
        h_layout = QHBoxLayout()
        button_save=QPushButton("Add and Select") 
        h_layout.addWidget(button_save)
        self.layout.addLayout(h_layout)
        button_save.clicked.connect(lambda ch, : ModifierDialog.addModifier(self))
        self.layout.addWidget(QLabel(""))        
        self.layout.addWidget(self.buttonBox)

    def addModifier(self):        
        if (not self.name): return
        mod_info={"name":self.name.text(),"modifiers":self.modifiers.toPlainText()}
        ModifierData.list.append(mod_info)
        ModifierData.save(ModifierData)
        SDConfig.params.modifiers = self.modifiers.toPlainText()
        self.accept()

    def selectModifier(self,num):
        mod_info=ModifierData.list[num]
        SDConfig.params.modifiers = mod_info["modifiers"]
        self.accept()

    def deleteModifier(self,num):
        qm = QMessageBox()
        ret = qm.question(self,'', "Are you sure?", qm.Yes | qm.No)
        if ret == qm.Yes:
            del ModifierData.list[num]
            ModifierData.save(ModifierData)
            self.accept()

    def modifierInput(self,layout):
        layout.addWidget(QLabel("Modifiers"))
        modifiers=QPlainTextEdit()
        modifiers.setPlainText(SDConfig.params.modifiers)      
        layout.addWidget(modifiers)
        h_layout = QHBoxLayout()
        button_presets=QPushButton("Presets...") 
        h_layout.addWidget(button_presets)
        button_copy_prompt=QPushButton("Copy full Prompt") 
        h_layout.addWidget(button_copy_prompt)        
        layout.addLayout(h_layout)
        button_presets.clicked.connect(lambda ch, : ModifierDialog.openModifierPresets(self))
        button_copy_prompt.clicked.connect(lambda ch, : ModifierDialog.copyPrompt(self))
        return modifiers        

    def copyPrompt(self):
        prompt=getFullPrompt(self)
        QApplication.clipboard().setText(prompt)

    def openModifierPresets(self):
        SDConfig.params.modifiers = self.modifiers.toPlainText()
        dlg=ModifierDialog()
        if dlg.exec():
            self.modifiers.setPlainText(SDConfig.params.modifiers)


dialogfields = {
    'txt2img':         ['prompt', 'negprompt', 'model', 'steps', 'scale', 'seed', 'batch', 'scheduler'],
    'img2img':         ['prompt', 'negprompt', 'model', 'strength', 'steps', 'scale', 'seed', 'batch', 'image', 'scheduler', 'prescale'],
    'upscale':         ['prompt', 'model', 'upscale_amount', 'scale', 'scheduler', 'image'],
    'inpaint':         ['prompt', 'negprompt', 'model', 'steps', 'scale', 'strength', 'seed', 'batch', 'image', 'scheduler', 'prescale'],
    'generateTiled':   ['prompt', 'negprompt', 'model', 'strength', 'scale', 'tile_method', 'tile_width', 'tile_height', 'tile_overlap', 'tile_alignmentx', 'tile_alignmenty', 'seed', 'scheduler', 'image'],
    'imagevariation':  ['steps', 'seed', 'scale', 'batch', 'image', 'scheduler'],
    'instructpix2pix': ['instruct', 'steps', 'scale', 'seed', 'batch', 'image', 'scheduler'],
    'preprocess':      ['model', 'image']
}


# default dialog for image generation: txt2img, img2img and inpainting
class SDDialog(QDialog):
    def __init__(self, action, images):
        super().__init__(None)
        self.config = SDConfig()
        self.action = action
        self.images = images
        self.setWindowTitle("Stable Diffusion " + action)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout = QHBoxLayout()
        formLayout= QVBoxLayout()
        self.layout.addLayout(formLayout)

        actionfields = dialogfields[action]
        modeltype = action

        if('prompt' in actionfields):
            formLayout.addWidget(QLabel("Prompt"))
            self.prompt = QPlainTextEdit()
            self.prompt.setPlainText(self.config.params.prompt)
            formLayout.addWidget(self.prompt)
            self.modifiers= ModifierDialog.modifierInput(self, formLayout)
            self.modifiers.setPlainText(self.config.params.modifiers)

        if('negprompt' in actionfields):
            formLayout.addWidget(QLabel("Negative"))
            self.negprompt = QLineEdit()
            self.negprompt.setText(self.config.params.negprompt)
            formLayout.addWidget(self.negprompt) 

        if('tile_method' in actionfields):
            tilemethod_label = QLabel("Tile method")
            formLayout.addWidget(tilemethod_label)
            self.tile_method = QComboBox()
            self.tile_method.addItems(['singlepass', 'multipass', 'inpaint'])
            tilemethod = self.config.params.tilemethod
            self.tile_method.setCurrentText(tilemethod)
            if(tilemethod in ('singlepass', 'multipass')):
                modeltype = 'img2img'
            else:
                modeltype = 'inpaint'
            self.tile_method.currentIndexChanged.connect(self.tileMethodChanged)
            formLayout.addWidget(self.tile_method)

        if('model' in actionfields):
            formLayout.addWidget(QLabel("Model"))
            self.model = QComboBox()
            models = getModels(modeltype)
            self.model.addItems([""] + models)
            if(action == "upscale"):
                self.model.setCurrentText(self.config.params.modelUpscale)
            elif(action == "preprocess"):
                self.model.setCurrentText(self.config.params.modelPreprocess)
            else:
                self.model.setCurrentText(self.config.params.modelGeneration)
            formLayout.addWidget(self.model)

        if('upscale_amount' in actionfields):
            upscale_label = QLabel("Upscale amount")
            formLayout.addWidget(upscale_label)
            self.upscale_amount, _ = self.addSlider(formLayout, self.config.params.upscaleamount, 2,4,1,1)

        if('tile_method' in actionfields):
            tilewidth_label = QLabel("Tile width")
            formLayout.addWidget(tilewidth_label)
            self.tile_width=createSlider(self, formLayout, self.config.params.tilewidth, 256, 1024, 64, 1)
            tileheight_label = QLabel("Tile height")
            formLayout.addWidget(tileheight_label)
            self.tile_height=createSlider(self, formLayout, self.config.params.tileheight, 256, 1024, 64, 1)
            tileoverlap_label = QLabel("Tile overlap")
            formLayout.addWidget(tileoverlap_label)
            self.tile_overlap = createSlider(self, formLayout, self.config.params.tileoverlap, 0, 384, 2, 1)

            tilealignmentx_label = QLabel("Tile alignment x")
            formLayout.addWidget(tilealignmentx_label)
            self.tile_alignmentx = QComboBox()
            self.tile_alignmentx.addItems(['tile_centre', 'tile_edge'])
            self.tile_alignmentx.setCurrentText(self.config.params.tilealignmentx)
            formLayout.addWidget(self.tile_alignmentx)

            tilealignmenty_label = QLabel("Tile alignment y")
            formLayout.addWidget(tilealignmenty_label)
            self.tile_alignmenty = QComboBox()
            self.tile_alignmenty.addItems(['tile_centre', 'tile_edge'])
            self.tile_alignmenty.setCurrentText(self.config.params.tilealignmenty)
            formLayout.addWidget(self.tile_alignmenty)

        if('strength' in actionfields):
            self.strength_label = QLabel("Strength")
            formLayout.addWidget(self.strength_label)
            self.strength, self.strength_value = self.addSlider(formLayout,self.config.params.strength*100,0,100,1,100)

        if('steps' in actionfields):
            self.steps_label=QLabel("Steps")
            formLayout.addWidget(self.steps_label)        
            self.steps, self.steps_value = self.addSlider(formLayout,self.config.params.steps,1,250,5,1)

        if('scale' in actionfields):
            scale_label=QLabel("Guidance Scale")
            scale_label.setToolTip("how strongly the image should follow the prompt")
            formLayout.addWidget(scale_label)        
            self.scale, _ = self.addSlider(formLayout,self.config.params.cfgscale*10,10,300,5,10)
     
        if('seed' in actionfields):
            seed_label=QLabel("Seed (empty=random)")
            seed_label.setToolTip("same seed and same prompt = same image")
            formLayout.addWidget(seed_label)      
            self.seed = QLineEdit()
            self.seed.setText(self.config.params.seed)
            formLayout.addWidget(self.seed)

        if('batch' in actionfields):
            formLayout.addWidget(QLabel("Number images"))        
            self.batch, _ = self.addSlider(formLayout, self.config.params.batch,1,4,1,1)
   
        if('scheduler' in actionfields):
            scheduler_label=QLabel("Scheduler")
            formLayout.addWidget(scheduler_label)           
            self.scheduler = QComboBox()
            self.scheduler.addItems([
                'DDIMScheduler',
                'DPMSolverMultistepScheduler', 
                'EulerAncestralDiscreteScheduler',
                'EulerDiscreteScheduler',
                # 'HeunDiscreteScheduler',
                # 'KDPM2AncestralDiscreteScheduler',
                # 'KDPM2DiscreteScheduler', 
                'LMSDiscreteScheduler', 
                'UniPCMultistepScheduler'
            ])
            self.scheduler.setCurrentText(self.config.params.scheduler)
            formLayout.addWidget(self.scheduler)

        if('prescale' in actionfields):
            prescale_label = QLabel("Prescale")
            formLayout.addWidget(prescale_label)  
            self.prescale = QComboBox()
            self.prescale.addItems(['0.5', '1.0', '2.0'])
            self.prescale.setCurrentText(str(self.config.params.prescale))
            formLayout.addWidget(self.prescale)

        formLayout.addWidget(QLabel(""))        
        formLayout.addWidget(self.buttonBox)

        if('image' in actionfields):
            control_models = getModels("control")
            self.controlmodels = [IMAGETYPE_INITIMAGE] + control_models
            self.control_model_dropdowns = []
            tabs = QTabWidget()
            for i, image in enumerate(images):
                self.addImageTab(tabs, image, i)
            self.layout.addWidget(tabs)
            self.controlModelChanged(0)

        self.setLayout(self.layout)


    def tileMethodChanged(self, index):
        if(self.tile_method.currentText() in ('singlepass', 'multipass')):
            modeltype = 'img2img'
        else:
            modeltype = 'inpaint'
        models = getModels(modeltype)
        # update itesm in model dropdown
        self.model.clear()
        self.model.addItems([""] + models)


    def addImageTab(self, tabs, image, i):
        tabWidget = QWidget()
        tabLayout = QVBoxLayout()      
        control_model_dropdown = QComboBox()
        control_model_dropdown.addItems(self.controlmodels)
        control_model_dropdown.currentIndexChanged.connect(self.controlModelChanged)
        controlimageparams = self.config.params.controlimages
        if i < len(controlimageparams):
            if controlimageparams[i].type == IMAGETYPE_INITIMAGE:
                control_model_dropdown.setCurrentText(IMAGETYPE_INITIMAGE)
            control_model_dropdown.setCurrentText(controlimageparams[i].model)
        self.control_model_dropdowns.append(control_model_dropdown)
        imgLabel = QLabel()
        imgLabel.setPixmap(self.maxSizePixmap(image, (896, 896)))
        tabLayout.addWidget(control_model_dropdown)
        tabLayout.addWidget(imgLabel)
        tabWidget.setLayout(tabLayout)
        tabs.addTab(tabWidget, f"Image {i}")

    
    def controlModelChanged(self, index):
        # TODO no need to change action, just put strength slider in the image tab
        action = "txt2img"
        for i, control_model_dropdown in enumerate(self.control_model_dropdowns):
            if (control_model_dropdown.currentText() == IMAGETYPE_INITIMAGE):
                action = "img2img"
                break
        print("controlModelChanged", action)
        if (hasattr(self, 'strength') and hasattr(self, 'steps')):
            self.steps.setVisible(action == "txt2img")
            self.steps_label.setVisible(action == "txt2img")
            self.steps_value.setVisible(action == "txt2img")
            self.strength.setVisible(action == "img2img")
            self.strength_label.setVisible(action == "img2img")
            self.strength_value.setVisible(action == "img2img")


    def maxSizePixmap(self, image, max_size):
        if image.width() > max_size[0] or image.height() > max_size[1]:
            return QPixmap.fromImage(image).scaled(max_size[0], max_size[1], Qt.KeepAspectRatio)
        else:
            return QPixmap.fromImage(image)

    # TODO replace with common createSlider 
    def addSlider(self,layout,value,min,max,steps,divider):
        h_layout = QHBoxLayout()
        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.setRange(min, max)

        slider.setSingleStep(steps)
        slider.setPageStep(steps)
        slider.setTickInterval
        label = QLabel(str(value), self)
        h_layout.addWidget(slider, stretch=9)
        h_layout.addWidget(label, stretch=1)
        if (divider!=1):
            slider.valueChanged.connect(lambda: slider.setValue(slider.value()//steps*steps) or label.setText( str(slider.value()/divider)))
        else:
            slider.valueChanged.connect(lambda: slider.setValue(slider.value()//steps*steps) or label.setText( str(slider.value())))
        slider.setValue(int(value))
        layout.addLayout(h_layout)
        return slider, label
        
    # put data from dialog in configuration and save it
    def saveParams(self):
        genParams = GenerationParameters()
        genParams.generationtype = self.action
        actionfields = dialogfields[self.action]
        if('prompt' in actionfields):
            self.config.params.prompt = self.prompt.toPlainText()
            self.config.params.modifiers = self.modifiers.toPlainText()
            genParams.prompt = getFullPrompt(self)
        if('negprompt' in actionfields):
            self.config.params.negprompt = self.negprompt.text()
            genParams.negprompt = self.negprompt.text()
        if('seed' in actionfields):
            if(self.seed.text() != ""):
                self.config.params.seed = int(self.seed.text())
                genParams.seed = int(self.seed.text())
            else:
                self.config.params.seed = None
                genParams.seed = None
        if('scale' in actionfields):
            self.config.params.cfgscale = self.scale.value()/10
            genParams.cfgscale = self.scale.value()/10
        if('scheduler' in actionfields):
            self.config.params.scheduler = self.scheduler.currentText()
            genParams.scheduler = self.scheduler.currentText()
        if('model' in actionfields):
            if(self.action == "upscale"):
                self.config.params.modelUpscale = self.model.currentText()
                genParams.upscalemethod = self.model.currentText().split("/")[0]
                genParams.models = [ModelParameters(name=self.model.currentText().split("/")[1])]
            elif(self.action == "preprocess"):
                self.config.params.modelPreprocess = self.model.currentText()
                genParams.models = [ModelParameters(name=self.model.currentText())]
            else:
                self.config.params.modelGeneration = self.model.currentText()
                genParams.models = [ModelParameters(name=self.model.currentText())]
        if('batch' in actionfields):
            self.config.params.batch = int(self.batch.value())
            genParams.batch = int(self.batch.value())
        if('strength' in actionfields):
            self.config.params.strength = self.strength.value()/100
            genParams.strength = self.strength.value()/100
        if('steps' in actionfields):
            self.config.params.steps = int(self.steps.value())
            genParams.steps = int(self.steps.value())
        if('tile_method' in actionfields):
            self.config.params.tilemethod = self.tile_method.currentText()
            genParams.tilemethod = self.tile_method.currentText()
            self.config.params.tilewidth = self.tile_width.value()
            genParams.tilewidth = self.tile_width.value()
            self.config.params.tileheight = self.tile_height.value()
            genParams.tileheight = self.tile_height.value()
            self.config.params.tileoverlap = self.tile_overlap.value()
            genParams.tileoverlap = self.tile_overlap.value()
            self.config.params.tilealignmentx = self.tile_alignmentx.currentText()
            genParams.tilealignmentx = self.tile_alignmentx.currentText()
            self.config.params.tilealignmenty = self.tile_alignmenty.currentText()
            genParams.tilealignmenty = self.tile_alignmenty.currentText()
        if('upscale_amount' in actionfields):
            self.config.params.upscaleamount = int(self.upscale_amount.value())
            genParams.upscaleamount = int(self.upscale_amount.value())
        if('prescale' in actionfields):
            self.config.params.prescale = self.prescale.currentText()
            genParams.prescale = float(self.prescale.currentText())
        if('image' in actionfields):
            images64 = base64EncodeImages(self.images)
            self.config.params.controlimages = []
            genParams.controlimages = []
            for i, control_model_dropdown in enumerate(self.control_model_dropdowns):
                if (control_model_dropdown.currentText() == IMAGETYPE_INITIMAGE):
                    self.config.params.controlimages.append(ControlImageParameters(type = IMAGETYPE_INITIMAGE))
                    genParams.controlimages.append(ControlImageParameters(type = IMAGETYPE_INITIMAGE, image64 = images64[i]))
                else:
                    self.config.params.controlimages.append(ControlImageParameters(type = IMAGETYPE_CONTROLIMAGE, model = control_model_dropdown.currentText()))
                    genParams.controlimages.append(ControlImageParameters(type = IMAGETYPE_CONTROLIMAGE, model = control_model_dropdown.currentText(), image64 = images64[i]))
        self.config.save()
        return genParams


# put image in Krita on new layer or existing one
def selectImage(params:GenerationParameters, qImg):  
    doc = getDocument()
    selection = doc.selection()        
    root = doc.rootNode()
    layer = doc.createNode(params.prompt, "paintLayer")
    root.addChildNode(layer, None)

    ptr = qImg.bits()
    ptr.setsize(qImg.byteCount())

    if (params.generationtype == "upscale" or params.generationtype == "face_enhance"):
        layer.setPixelData(QByteArray(ptr.asstring()),0,0,qImg.width(),qImg.height())
        #resize canvas to fit
        if(qImg.width() > doc.width()):
            doc.setWidth(qImg.width())
        if(qImg.height() > doc.height()):
            doc.setHeight(qImg.height())
        # TODO set scale annotation doc.setAnnotation("scale", );
    else:
        if(selection == None):
            layer.setPixelData(QByteArray(ptr.asstring()),0,0,qImg.width(),qImg.height())
        else:
            layer.setPixelData(QByteArray(ptr.asstring()),selection.x(),selection.y(),qImg.width(),qImg.height())

    doc.waitForDone()
    doc.refreshProjection() 


# asking for image of result set and update option
class ImageResutDialog(QDialog):
    def __init__(self, images, params: GenerationParameters):
        super().__init__(None)
        self.config = SDConfig()
        self.images=images
        self.setWindowTitle("Result")
        QBtn = QDialogButtonBox.Cancel
        self.genparam=params
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        top_layout=QVBoxLayout()        
        prompt_layout=QHBoxLayout()        
        top_layout.addLayout(prompt_layout)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlainText(self.config.params.prompt)
        prompt_layout.addWidget(self.prompt,stretch=9)
        btn_regenerate=QPushButton("Generate with steps "+str(self.config.params.steps))      
        btn_regenerate.clicked.connect(self.regenerateStart)
        prompt_layout.addWidget(btn_regenerate,stretch=1)
        self.modifiers= ModifierDialog.modifierInput(self,top_layout)

        layout=QHBoxLayout()
        top_layout.addLayout(layout)
        self.imgLabels=[0]*len(images)
        self.seedLabel=[0]*len(images)
        i=0
        for imagedata in images:       
            v_layout = QVBoxLayout()
            layout.addLayout(v_layout)       
            imgLabel=QLabel()
            v_layout.addWidget(imgLabel) 
            imgLabel.setPixmap(QPixmap.fromImage(imagedata["qimage"]).scaled(384,384,Qt.KeepAspectRatio))  
            seedLabel=QLabel(str(imagedata.get("seed", -1)))
            seedLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.seedLabel[i]=seedLabel
            v_layout.addWidget(seedLabel)     
            self.imgLabels[i]=imgLabel
            h_layout = QHBoxLayout()
            v_layout.addLayout(h_layout)               
            btn=QPushButton("Select")          
            h_layout.addWidget(btn,stretch=1)
            btn.clicked.connect(lambda ch, num=i: selectImage(params, self.images[num]["qimage"]))
            btnUpdate=QPushButton()  
            btnUpdate.setIcon( Krita.instance().icon('updateColorize'))        
            h_layout.addWidget(btnUpdate,stretch=1)
            btnUpdate.clicked.connect(lambda ch, num=i: self.updateImageStart(num))
            i=i+1
            
        #TODO add scale control here
        
        top_layout.addWidget(QLabel("Update one image with new Steps value"))
        self.steps_update, _ = SDDialog.addSlider(self,top_layout,self.config.params.steps,1,250,5,1)
        
        if (params.generationtype in ("img2img", "inpaint", "generateTiled")):
            top_layout.addWidget(QLabel("Update with new Strengths value"))
            self.strength_update, _ = SDDialog.addSlider(self,top_layout,self.config.params.strength*100,0,100,1,100)

        self.setLayout(top_layout)

    # start request for HQ version of one image
    def regenerateStart(self):
        params = copy(self.SDParam)
        #SDConfig.dlgData["steps_update"]=self.steps_update.value()
        #p.steps=SDConfig.dlgData["steps_update"]
        self.config.params.prompt = self.prompt.text()
        self.config.params.modifiers = self.modifiers.toPlainText()
        self.config.save()
        params.prompt= getFullPrompt(self)        
        params.imageDialog=self
        params.regenerate=True
        images = runSD(params)
        # TODO display new image
        #     if (params.regenerate):
        #         print("generate new")
        #         params.imageDialog.updateImages(images)
        #     else:  
        #         params.imageDialog.updateImage(images[0])

    def updateImages(self, images):
        i=0
        self.images=images
        for image in images:
            imgLabel=self.imgLabels[i]
            seedLabel=self.seedLabel[i]
            seedLabel.setText(image["seed"])
            imgLabel.setPixmap(QPixmap.fromImage(image["qimage"]).scaled(384,384,Qt.KeepAspectRatio))  
            i=i+1

    # update one single image with new parameters
    def updateImageStart(self,num):
        params = copy(self.genparams)
        params.seed=self.images[num]["seed"]
        if (params.action in ("img2img", "inpaint")): 
            params.strength = self.strength_update.value()/100
        self.config.params.steps = self.steps_update.value()
        self.config.save()
        params.batch=1
        params.steps=self.config.params.steps
        self.config.params.prompt = self.prompt.toPlainText()
        self.config.params.modifiers = self.modifiers.toPlainText()
        params.prompt = getFullPrompt(self)
        self.updateImageNum=num
        params.imageDialog=self
        runSD(params)

    # update image with HQ version       
    def updateImage(self, imagedata):
        num=self.updateImageNum
        imgLabel=self.imgLabels[num]
        self.images[num]["qimage"]=imagedata["qimage"]
        imgLabel.setPixmap(QPixmap.fromImage(imagedata["qimage"]).scaled(384,384,Qt.KeepAspectRatio))  


def showImageResultDialog(imagedata,params):
    dlg = ImageResutDialog(imagedata,params)
    if dlg.exec():
        print("HQ Update here")

 
 # convert image from server result into QImage
def base64ToQImage(data):
  #   data=data.split(",")[1] # get rid of data:image/png,
     image64 = data.encode('ascii')
     imagen = QtGui.QImage()
     bytearr = QtCore.QByteArray.fromBase64( image64 )
     imagen.loadFromData( bytearr, 'PNG' )      
     return imagen


def getModels(type) -> List[str]:
    config = SDConfig()
    # TODO build into models endpoint
    if(type == "upscale"):
        return ["esrgan/remacri", 
                "esrgan/lollipop", 
                "esrgan/ultrasharp"]
    if(type == "preprocess"):
        return ['DepthEstimation',
                'NormalEstimation',
                'CannyEdge',
                'HEDEdgeDetection',
                'PIDIEdgeDetection',
                'MLSDStraightLineDetection',
                'PoseDetection',
                'Segmentation',
                'ContentShuffle']
    endpoint=config.url
    endpoint=endpoint.strip("/")
    endpoint+=f"/api/models?type={type}"
    headers = {
        "Accept": "application/json",
    } 
    req = urllib.request.Request(endpoint, None, headers, method="GET")
    with urllib.request.urlopen(req) as f:
        res = f.read()
    models = json.loads(res)
    modelids = [model["modelid"] for model in models]
    modelids.sort()
    return modelids


def runSD(params:GenerationParameters, asynchronous=True):
    #TODO use generate and generateTiled instead of img2img and txt2img in backend
    if(params.generationtype == "img2img" or params.generationtype == "txt2img"):
        action = "generate"
    elif(params.generationtype == "generateTiled"):
        action = "generateTiled"
        params.generationtype = "img2img"
    else:
        action = params.generationtype

    if (asynchronous):
        res=getServerDataAsync(action, params)
    else:
        res=getServerData(action, params)

    if not res: 
        return    
    response=json.loads(res)
    # print(response)
    images = []
    seeds=[]

    for image in response["images"]:
        image["qimage"] = base64ToQImage(image["image"])
        images.append(image)
        if ("seed" in image):
            seeds.append(image["seed"])

    return images
   

def getFullPrompt(dlg):
    modifiers=""
    list=dlg.modifiers.toPlainText().split("\n")
    for i in range(0,len(list)):
        m=list[i]
        if (m and m[0]!="#"): modifiers+=", "+m

   # modifiers=dlg.modifiers.toPlainText().replace("\n", ", ")
    prompt=dlg.prompt.toPlainText()
    if (not prompt):      
        errorMessage("Empty prompt","Type some text in prompt input box about what you want to see.")
        return ""
    prompt+=modifiers
    return prompt

def TxtToImage():
    s=getSelection()
    if (s==None):
        return
    dlg = SDDialog("txt2img",None)
    dlg.resize(700,200)
    if dlg.exec():
        params = dlg.saveParams()
        images = runSD(params)
        showImageResultDialog(images, params)


def ImageToImage():
    images = getLayerSelections()
    dlg = SDDialog("img2img",images)
    dlg.resize(900,200)
    if dlg.exec():
        params = dlg.saveParams()
        images = runSD(params)
        showImageResultDialog(images, params)


def TiledImageToImage():
    images = getLayerSelections()
    dlg = SDDialog("generateTiled", images)
    dlg.resize(900,200)
    if dlg.exec():
        params = dlg.saveParams()
        images = runSD(params)
        showImageResultDialog(images, params)


def Upscale(): 
    images = getLayerSelections()
    dlg = SDDialog("upscale", images)
    dlg.resize(900,200)

    if dlg.exec():
        params = dlg.saveParams()
        images = runSD(params)
        showImageResultDialog(images, params)


def Preprocess():
    images = getLayerSelections()
    dlg = SDDialog("preprocess", images)
    dlg.resize(900,200)
    if dlg.exec():
        params = dlg.saveParams()
        images = runSD(params)
        showImageResultDialog(images, params)


def InstructPixToPix():
    images = getLayerSelections()
    dlg = SDDialog("instructpix2pix", images)
    dlg.resize(900,200)
    if dlg.exec():
        params = dlg.saveParams()
        images = runSD(params)
        showImageResultDialog(images, params)


def Inpaint():    
    images = getLayerSelections()
    imageWithTransparency = images[0]
    foundTrans=False
    foundPixel=False
    for i in range(imageWithTransparency.width()):
        for j in range(imageWithTransparency.height()):
            rgb = imageWithTransparency.pixel(i, j)
            alpha = qAlpha(rgb)
            if (alpha !=255):
                foundTrans=True
            else: 
                foundPixel=True
    if (foundTrans==False):
        errorMessage("No transparent pixels found","Needs content with part removed by eraser (Brush in Tool palette + Right click for Eraser selection)")
        return
    if (foundPixel==False):
        errorMessage("No  pixels found","Maybe wrong layer selected? Choose one with some content n it.")
        return
    dlg = SDDialog("inpaint",images) 
    dlg.resize(900,200)
    if dlg.exec():
        params = dlg.saveParams()
        images = runSD(params)
        showImageResultDialog(images, params)


# config dialog
def Config():
    dlg=SDConfigDialog()
    if dlg.exec():
        dlg.save()


# expand selection to max size        
def expandSelection():
    config = SDConfig()
    d = getDocument()

    if (d==None): return
    s = d.selection()    
    if (not s):  x=0;y=0
    else: x=s.x(); y=s.y()     
    s2 = Selection()    
    s2.select(x, y, config.width, config.height, 1)
    d.setSelection(s2)
    d.refreshProjection()
