from cmd import PROMPT
from multiprocessing import dummy
import urllib.request
import http.client
import json
from krita import *
from PyQt5.Qt import QByteArray
from PyQt5.QtGui  import QImage, QPixmap
import array
from copy import copy
from pathlib import Path

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
        str=self.serialize(self)
        Krita.instance().writeSetting ("SDPlugin", "Modifiers", str)

#        with open(rPath / "krita_ai_modifiers.config", 'w', encoding='utf-8') as f_out:#
#            f_out.write(str)
    def load(self):
        str=Krita.instance().readSetting ("SDPlugin", "Modifiers",None)
        if (not str): return
        self.unserialize(self,str)    

#        if (not (rPath / "krita_ai_modifiers.config").exists()): return
#        with open(rPath / "krita_ai_modifiers.config", 'r', encoding='utf-8') as f_in:
 #            str=f_in.read()
  #      self.unserialize(self,str)    

class SDConfig:
    "This is Stable Diffusion Plugin Main Configuration"     
    url = "http://localhost:5000"
    type="Colab"
    inpaint_mask_blur=4
    inpaint_mask_content="latent noise"     
    width=512
    height=512    
    dlgData={
        "action": "txt2img",
        "prompt": "",
        "negprompt": "",
        "seed": "",
        "steps": 15,
        "steps_update": 50,
        "num": 2,
        "modifiers": "highly detailed\n",
        "scale": 7.5,
        "strength": .75,
        "scheduler":"DPMSolverMultistepScheduler",
        "upscale_amount": 2,
        "upscale_method": "all"
    }


    def serialize(self):
        obj={
            "url":self.url,
            "type":self.type,
            "inpaint_mask_blur":self.inpaint_mask_blur, 
            "inpaint_mask_content":self.inpaint_mask_content,
            "width":self.width, 
            "height":self.height,
            "type": self.type,
            "dlgData":self.dlgData
        }
        return json.dumps(obj)
    def unserialize(self,str):
        obj=json.loads(str)
        self.url=obj.get("url","http://localhost:7860")
        self.type=obj.get("type","Colab")
        self.dlgData=obj["dlgData"]
        self.inpaint_mask_blur=obj.get("inpaint_mask_blur",4)
        self.inpaint_mask_content=obj.get("inpaint_mask_content","latent noise")
        self.width=obj.get("width",512)
        self.height=obj.get("height",512)
    def save(self):
        str=self.serialize(self)
        Krita.instance().writeSetting ("SDPlugin", "Config", str)
    def load(self):
        str=Krita.instance().readSetting ("SDPlugin", "Config",None)
        if (not str): return
        self.unserialize(self,str)

SDConfig.load(SDConfig)

class SDParameters:
    "This is Stable Diffusion Parameter Class"     
    prompt = ""
    negprompt = ""
    steps = 0
    seed = 0
    num =0
    strength=1.0
    scale=0.5
    seedList =["","","",""]
    imageDialog = None
    regenerate = False
    image64=""
    maskImage64=""
    scheduler="DPMSolverMultistepScheduler"
    inpaint_mask_blur=4
    inpaint_mask_content="latent noise" 
    action="txt2img"
    strength = 1 
    upscale_amount = 1
    upscale_method = "all"

def errorMessage(text,detailed):
    msgBox= QMessageBox()
    msgBox.resize(500,200)
    msgBox.setWindowTitle("Stable Diffusion")
    msgBox.setText(text)
    msgBox.setDetailedText(detailed)
    msgBox.setStyleSheet("QLabel{min-width: 700px;}")
    msgBox.exec()


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
        self.setWindowTitle("Stable Diffusion Configuration")
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout = QVBoxLayout()
        link_label=QLabel('Webservice URL<br>\nYou need running <a href="https://colab.research.google.com/drive/1eq-VF2dRHgxNjtPX5iZbDPuwtfJIEhtx#scrollTo=bYpxm4Fw2tLT">Colab with new API</a><br>\nCheck if web interface is working there before you use this plugin. Use http://xxxx.ngrok.io link here.')
        link_label.setOpenExternalLinks(True)
        self.layout.addWidget(link_label)
        self.url = QLineEdit()
        self.url.setText(SDConfig.url)    
        self.layout.addWidget(self.url)
        self.layout.addWidget(QLabel('For local version you need this fork running <br> <a href="https://github.com/imperator-maximus/stable-diffusion-webui">imperator-maximus/stable-diffusion-webui</a><br>\n start api.bat there'))

        self.layout.addWidget(QLabel(''))
        inpainting_label=QLabel('Inpainting options')
        inpainting_label.setToolTip('You can play around with these two values. Default is 4 and "latent noise"')
        self.layout.addWidget(inpainting_label)
        h_layout_inpaint=QHBoxLayout()

        self.inpaint_mask_blur=QLineEdit()
        self.inpaint_mask_blur.setText(str(SDConfig.inpaint_mask_blur))
        h_layout_inpaint.addWidget(QLabel('Mask Blur:'),stretch=1)
        h_layout_inpaint.addWidget(self.inpaint_mask_blur,stretch=1)
        h_layout_inpaint.addWidget(QLabel('Masked Content:'),stretch=1)
        self.inpaint_mask_content = QComboBox()
        self.inpaint_mask_content.addItems(['fill', 'original', 'latent noise', 'latent nothing','g-diffusion'])
        self.inpaint_mask_content.setCurrentText(SDConfig.inpaint_mask_content)
        h_layout_inpaint.addWidget(self.inpaint_mask_content,stretch=1)      
        h_layout_inpaint.addWidget(QLabel(''),stretch=5)

        self.layout.addLayout(h_layout_inpaint)

        self.layout.addWidget(QLabel(''))
        self.layout.addWidget(QLabel('Select tool size'))

        h_layout_width=QHBoxLayout()
        h_layout_width.addWidget(QLabel('Width:'))
        self.width=createSlider(self, h_layout_width, SDConfig.width, 256, 1024, 64, 1)      
        h_layout_width.addWidget(self.width)
        self.layout.addLayout(h_layout_width)

        h_layout_height=QHBoxLayout()
        h_layout_height.addWidget(QLabel('Height:'))
        self.height=createSlider(self, h_layout_height, SDConfig.height, 256, 1024, 64, 1)
        h_layout_height.addWidget(self.height)
        self.layout.addLayout(h_layout_height)

        self.layout.addWidget(QLabel(''))
        self.layout.addWidget(QLabel(''))

        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        self.resize(500,200)

    def save(self):
        SDConfig.url=self.url.text()
        SDConfig.inpaint_mask_blur=int(self.inpaint_mask_blur.text())
        SDConfig.inpaint_mask_content=self.inpaint_mask_content.currentText()
        SDConfig.width=self.width.value()
        SDConfig.height=self.height.value()
        SDConfig.save(SDConfig)

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
        self.modifiers.setPlainText(SDConfig.dlgData.get("modifiers",""))      
        self.layout.addWidget(self.modifiers)     

        self.layout.addWidget(QLabel("Example Prompt"))
        self.example_prompt = QLineEdit()
        self.example_prompt.setText(SDConfig.dlgData.get("prompt",""))       
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
        SDConfig.dlgData["modifiers"]=self.modifiers.toPlainText()
        self.accept()
    def selectModifier(self,num):
        mod_info=ModifierData.list[num]
        SDConfig.dlgData["modifiers"]=mod_info["modifiers"]
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
        modifiers.setPlainText(SDConfig.dlgData.get("modifiers",""))      
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
        SDConfig.dlgData["modifiers"]=self.modifiers.toPlainText()
        dlg=ModifierDialog()
        if dlg.exec():            
            self.modifiers.setPlainText(SDConfig.dlgData["modifiers"])


# default dialog for image generation: txt2img, img2img and inpainting
class SDDialog(QDialog):
    def __init__(self,action,image):
        super().__init__(None)
        SDConfig.dlgData["action"]=action
        data=SDConfig.dlgData

        self.setWindowTitle("Stable Diffusion "+data["action"])

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout = QHBoxLayout()
        formLayout= QVBoxLayout()
        self.layout.addLayout(formLayout)

        formLayout.addWidget(QLabel("Prompt"))
        self.prompt = QLineEdit()
        self.prompt.setText(data["prompt"])            
        formLayout.addWidget(self.prompt)
        self.modifiers= ModifierDialog.modifierInput(self,formLayout)

        if (data["action"] not in ("upscale")):
            formLayout.addWidget(QLabel("Negative"))
            self.negprompt = QLineEdit()
            self.negprompt.setText(data.get("negprompt", ""))
            formLayout.addWidget(self.negprompt) 

        if (data["action"] in ("upscale")):
            upscalemethod_label=QLabel("Upscale method")
            formLayout.addWidget(upscalemethod_label)
            self.upscale_method = QComboBox()
            self.upscale_method.addItems(['laczos3', 'esrgan/lollypop', 'esrgan/remacri', 'stable-diffusion'])
            self.upscale_method.setCurrentText(data.get("upscale_method","all"))
            formLayout.addWidget(self.upscale_method)
            upscale_label = QLabel("Upscale amount")
            formLayout.addWidget(upscale_label)
            self.upscale_amount=self.addSlider(formLayout, data.get("upscale_amount", 2), 2,4,1,1)

        if (data["action"] in ("img2img", "img2imgTiled")):
            formLayout.addWidget(QLabel("Strength"))
            self.strength=self.addSlider(formLayout,data["strength"]*100,0,100,1,100)

        if (data["action"] in ("txt2img", "inpaint")):
            steps_label=QLabel("Steps")
            steps_label.setToolTip("more steps = slower but often better quality. Recommendation start with lower step like 15 and update in image overview with higher one like 50")
            formLayout.addWidget(steps_label)        
            self.steps=self.addSlider(formLayout,data["steps"],1,250,5,1)

        if (data["action"] not in ("upscale")):
            scale_label=QLabel("Guidance Scale")
            scale_label.setToolTip("how strongly the image should follow the prompt")
            formLayout.addWidget(scale_label)        
            self.scale=self.addSlider(formLayout,data.get("scale", 9)*10,10,300,5,10)
     
            seed_label=QLabel("Seed (empty=random)")
            seed_label.setToolTip("same seed and same prompt = same image")
            formLayout.addWidget(seed_label)      
            self.seed = QLineEdit()
            self.seed.setText(data["seed"])                  
            formLayout.addWidget(self.seed)

        if (not data["action"]=="upscale" and not data["action"]=="img2imgTiled"):
            formLayout.addWidget(QLabel("Number images"))        
            self.num=self.addSlider(formLayout,data["num"],1,4,1,1)
   
        scheduler_label=QLabel("Scheduler")
        formLayout.addWidget(scheduler_label)           
        self.scheduler = QComboBox()
        self.scheduler.addItems(['DPMSolverMultistepScheduler', 'EulerDiscreteScheduler', 'EulerAncestralDiscreteScheduler'])
        self.scheduler.setCurrentText(data.get("scheduler","DPMSolverMultistepScheduler"))
        formLayout.addWidget(self.scheduler)
        formLayout.addWidget(QLabel(""))        
        formLayout.addWidget(self.buttonBox)
        if (not data["action"]=="txt2img" and not data["action"]=="upscale" and not data["action"]=="img2imgTiled"):
            imgLabel=QLabel()        
            self.layout.addWidget(imgLabel) 
            imgLabel.setPixmap(QPixmap.fromImage(image))  
        self.setLayout(self.layout)

    # TODO replace with common createSlider 
    def addSlider(self,layout,value,min,max,steps,divider):
        h_layout =  QHBoxLayout()
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
        return slider
        
    # put data from dialog in configuration and save it        
    def setDlgData(self):

        if SDConfig.dlgData["action"] not in ("upscale"):
            SDConfig.dlgData["prompt"]=self.prompt.text()
            SDConfig.dlgData["negprompt"]=self.negprompt.text()
            SDConfig.dlgData["seed"]=self.seed.text()
            SDConfig.dlgData["scale"]=self.scale.value()/10
            SDConfig.dlgData["modifiers"]=self.modifiers.toPlainText()
            SDConfig.dlgData["scheduler"]=self.scheduler.currentText()
        
        if SDConfig.dlgData["action"] in ("txt2img", "img2img", "inpaint"):
            SDConfig.dlgData["num"]=int(self.num.value())
        if SDConfig.dlgData["action"] in ("img2img", "img2imgTiled"):
            SDConfig.dlgData["strength"]=self.strength.value()/100
        if SDConfig.dlgData["action"] in ("txt2img", "inpaint"):
            SDConfig.dlgData["steps"]=int(self.steps.value())
        if SDConfig.dlgData["action"] in ("upscale"):
            SDConfig.dlgData["upscale_amount"]=int(self.upscale_amount.value())
            SDConfig.dlgData["upscale_method"]=self.upscale_method.currentText()

        SDConfig.save(SDConfig)


# put image in Krita on new layer or existing one
def selectImage(params: SDParameters,qImg):  
    doc = getDocument()
    selection = doc.selection()        
    root = doc.rootNode()
    layer = doc.createNode(params.prompt, "paintLayer")
    root.addChildNode(layer, None)

    ptr = qImg.bits()
    ptr.setsize(qImg.byteCount())

    if (params.action == "upscale" or params.action == "face_enhance"):
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
class showImages(QDialog):
    def __init__(self, images, params: SDParameters):
        super().__init__(None)
        self.images=images
        self.setWindowTitle("Result")
        QBtn = QDialogButtonBox.Cancel
        self.SDParam=params
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        top_layout=QVBoxLayout()        
        prompt_layout=QHBoxLayout()        
        top_layout.addLayout(prompt_layout)
        self.prompt = QLineEdit()
        self.prompt.setText(SDConfig.dlgData.get("prompt",""))
        prompt_layout.addWidget(self.prompt,stretch=9)
        btn_regenerate=QPushButton("Generate with steps "+str(SDConfig.dlgData["steps"]))         
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
            imgLabel.setPixmap(QPixmap.fromImage(imagedata["qimage"]).scaled(380,380,Qt.KeepAspectRatio))  
            seedLabel=QLabel(str(imagedata["seed"]))
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
            
        #layout.addWidget(self.buttonBox)
        top_layout.addWidget(QLabel("Update one image with new Steps value"))
        self.steps_update=SDDialog.addSlider(self,top_layout,SDConfig.dlgData.get("steps_update",50),1,250,5,1)
        
        if (params.action in ("img2img", "inpaint")):
            top_layout.addWidget(QLabel("Update with new Strengths value"))
            self.strength_update=SDDialog.addSlider(self,top_layout,SDConfig.dlgData.get("strength",0.5)*100,0,100,1,100)

        self.setLayout(top_layout)

    # start request for HQ version of one image
    def regenerateStart(self):
        p = copy(self.SDParam)
        #SDConfig.dlgData["steps_update"]=self.steps_update.value()
        #p.steps=SDConfig.dlgData["steps_update"]
        #SDConfig.save(SDConfig)
        SDConfig.dlgData["prompt"]=self.prompt.text()
        SDConfig.dlgData["modifiers"]=self.modifiers.toPlainText()
        SDConfig.save(SDConfig)
        p.prompt= getFullPrompt(self)        
        p.imageDialog=self
        p.regenerate=True
        runSD(p)

    def updateImages(self, images):
        i=0
        self.images=images
        for image in images:
            imgLabel=self.imgLabels[i]
            seedLabel=self.seedLabel[i]
            seedLabel.setText(image["seed"])
            imgLabel.setPixmap(QPixmap.fromImage(image["qimage"]).scaled(380,380,Qt.KeepAspectRatio))  
            i=i+1

    # update one single image with new parameters
    def updateImageStart(self,num):
        p = copy(self.SDParam)
        p.seed=self.images[num]["seed"]
        if (p.action in ("img2img", "inpaint")): 
            p.strength=self.strength_update.value()/100
        SDConfig.dlgData["steps_update"]=self.steps_update.value()
        SDConfig.save(SDConfig)
        p.num=1
        p.steps=SDConfig.dlgData["steps_update"]
        SDConfig.dlgData["prompt"]=self.prompt.text()
        SDConfig.dlgData["modifiers"]=self.modifiers.toPlainText()
        p.prompt= getFullPrompt(self)        
        self.updateImageNum=num
        p.imageDialog=self
        runSD(p)

    # update image with HQ version       
    def updateImage(self, imagedata):
        num=self.updateImageNum
        imgLabel=self.imgLabels[num]
        self.images[num]["qimage"]=imagedata["qimage"]
        imgLabel.setPixmap(QPixmap.fromImage(imagedata["qimage"]).scaled(380,380,Qt.KeepAspectRatio))  


def imageResultDialog(imagedata,params):
    dlg = showImages(imagedata,params)
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

def getServerData(action, reqData):
    endpoint=SDConfig.url
    endpoint=endpoint.strip("/")
    endpoint+="/api/"+action
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }    
    try:
        print("endpoint")
        print(endpoint)
        req = urllib.request.Request(endpoint, reqData, headers)
        with urllib.request.urlopen(req) as f:
            res = f.read()
            return res
    except http.client.IncompleteRead as e:
        print("Incomplete Read Exception - better restart Colab or ")
        res = e.partial 
        return res           
    except Exception as e:
        error_message = traceback.format_exc() 
        errorMessage("Server Error","Endpoint: "+endpoint+", Reason: "+error_message)        
        return None


def runSD(params: SDParameters):
    # dramatic interface change needed!
    Colab=True
    if (SDConfig.type=="Local"): Colab=False
    if (not params.seed): seed=None
    else: seed=int(params.seed)
    inpainting_fill_options= ['fill', 'original', 'latent noise', 'latent nothing',"g-diffusion"]
    inpainting_fill=inpainting_fill_options.index(SDConfig.inpaint_mask_content)
    j = {'prompt': params.prompt, \
        'negprompt': params.negprompt, \
        'initimage': params.image64, \
        'maskimage': params.maskImage64, \
        'steps':params.steps, \
        'scheduler':params.scheduler, \
        'mask_blur': SDConfig.inpaint_mask_blur, \
        'inpainting_fill':inpainting_fill, \
        'use_gfpgan': False, \
        'batch': params.num, \
        'scale': params.scale, \
        'strength': params.strength, \
        'seed':seed, \
        'height':SDConfig.height, \
        'width':SDConfig.width, \
        'method': params.upscale_method, \
        'amount': params.upscale_amount, \
        'upscale_overlap':64, \
        'inpaint_full_res':True, \
        'inpainting_mask_invert': 0 \
        }    

    print(j)
    data = json.dumps(j).encode("utf-8")
    res=getServerData(params.action, data)
    if not res: return    
    response=json.loads(res)
    # print(response)
    num = len(response["images"])
    images = []
    params.seedList=[0]*num

    for image in response["images"]:
        image["qimage"] = base64ToQImage(image["image"])
        images.append(image)

    if (params.imageDialog):  # only refresh image
        if (params.regenerate):
            print("generate new")
            params.imageDialog.updateImages(images)
        else:  
            params.imageDialog.updateImage(images[0])
    else:
        imageResultDialog(images,params)
    return images


def getDocument():
    d = Application.activeDocument()
    if (d==None):  errorMessage("Please add a document","Needs document with a layer and selection.")
    return d

def getLayer():
    d=getDocument()
    if (d==None):  return
    n = d.activeNode()
    return n

def getSelection():
    d = getDocument()
    if (d==None): return
    s = d.selection()
    if (s==None):  errorMessage("Please make a selection","Operation runs on a selection only. Please use rectangle select tool.")
    return s      

def getFullPrompt(dlg):
    modifiers=""
    list=dlg.modifiers.toPlainText().split("\n")
    for i in range(0,len(list)):
        m=list[i]
        if (m and m[0]!="#"): modifiers+=", "+m

   # modifiers=dlg.modifiers.toPlainText().replace("\n", ", ")
    prompt=dlg.prompt.text()
    if (not prompt):      
        errorMessage("Empty prompt","Type some text in prompt input box about what you want to see.")
        return ""
    prompt+=modifiers
    return prompt

def TxtToImage():
    s=getSelection()
    if (s==None):   return   
    SDConfig.load(SDConfig)
    dlg = SDDialog("txt2img",None)
    dlg.resize(700,200)
    if dlg.exec():
        dlg.setDlgData()
        p = SDParameters()
        p.prompt=getFullPrompt(dlg)
        if not p.prompt: return        
        p.action="txt2img"
        data=SDConfig.dlgData
        p.negprompt = data["negprompt"]
        p.steps=data["steps"]
        p.seed=data["seed"]
        p.num=data["num"]
        p.scheduler=data["scheduler"]
        p.scale=data["scale"]
        runSD(p)


def base64EncodeImage(image):
    data = QByteArray()
    buf = QBuffer(data)
    image.save(buf, 'PNG')
    ba=data.toBase64()
    image64=str(ba,"ascii")
    return image64


def ImageToImage():
    s=getSelection()
    if (s==None):   
        return    
    n=getLayer()
    data=n.pixelData(s.x(),s.y(),s.width(),s.height())
    image=QImage(data.data(),s.width(),s.height(),QImage.Format_RGBA8888).rgbSwapped()
    image64 = base64EncodeImage(image)
    
    dlg = SDDialog("img2img",image)
    dlg.resize(900,200)

    if dlg.exec():
        dlg.setDlgData()
        p = SDParameters()
        p.prompt=getFullPrompt(dlg)
        if not p.prompt: return
        data=SDConfig.dlgData
        p.action="img2img"
        p.negprompt = data["negprompt"]
        p.steps=data["steps"]
        p.seed=data["seed"]
        p.num=data["num"]
        p.scale=data["scale"]
        p.scheduler=data["scheduler"]
        p.image64=image64
        p.strength=data["strength"]
        runSD(p)


def TiledImageToImage():
    doc = getDocument()
    layer = getLayer()
    selection = doc.selection()
    if(selection is None):
        data = layer.pixelData(0, 0, doc.width(), doc.height())
        image = QImage(data.data(),doc.width(),doc.height(),QImage.Format_RGBA8888).rgbSwapped()
    else:
        data=layer.pixelData(selection.x(), selection.y(), selection.width(), selection.height())
        image = QImage(data.data(), selection.width(), selection.height(), QImage.Format_RGBA8888).rgbSwapped()
    image64 = base64EncodeImage(image)
    
    dlg = SDDialog("img2imgTiled",image)
    dlg.resize(900,200)

    if dlg.exec():
        dlg.setDlgData()
        p = SDParameters()
        p.prompt=getFullPrompt(dlg)
        if not p.prompt: return
        data=SDConfig.dlgData
        p.action="img2imgTiled"
        p.negprompt = data["negprompt"]
        p.steps=data["steps"]
        p.seed=data["seed"]
        p.num=1
        p.scale=data["scale"]
        p.scheduler=data["scheduler"]
        p.image64=image64
        p.strength=data["strength"]
        runSD(p)


def Upscale(): 
    doc = getDocument()
    layer = getLayer()
    selection = doc.selection()
    if(selection is None):
        data = layer.pixelData(0, 0, doc.width(), doc.height())
        image = QImage(data.data(),doc.width(),doc.height(),QImage.Format_RGBA8888).rgbSwapped()
    else:
        data=layer.pixelData(selection.x(), selection.y(), selection.width(), selection.height())
        image = QImage(data.data(), selection.width(), selection.height(), QImage.Format_RGBA8888).rgbSwapped()
    image64 = base64EncodeImage(image)
    
    dlg = SDDialog("upscale", image)
    dlg.resize(900,200)

    if dlg.exec():
        dlg.setDlgData()
        params = SDParameters()
        params.prompt=getFullPrompt(dlg)
        # if not params.prompt: return
        data=SDConfig.dlgData
        params.action = "upscale"
        # params.steps = data["steps"]
        # params.seed = data["seed"]
        params.num = 1
        # params.scale = data["scale"]
        params.image64 = image64
        # params.strength = data["strength"]
        # p.scheduler=data["scheduler"]
        params.upscale_amount = data["upscale_amount"]
        params.upscale_method = data["upscale_method"]
        runSD(params)


def FaceEnhance(): 
    doc = getDocument()
    layer = getLayer()
    selection = doc.selection()
    if(selection is None):
        data = layer.pixelData(0, 0, doc.width(), doc.height())
        image = QImage(data.data(),doc.width(),doc.height(),QImage.Format_RGBA8888).rgbSwapped()
    else:
        data=layer.pixelData(selection.x(), selection.y(), selection.width(), selection.height())
        image = QImage(data.data(), selection.width(), selection.height(), QImage.Format_RGBA8888).rgbSwapped()
    image64 = base64EncodeImage(image)
    
    params = SDParameters()
    params.action = "face_enhance"
    params.image64 = image64
    params.num = 1
    params.scale = None
    params.prompt = "Face Enhance"
    runSD(params)


def Inpaint():    
    n = getLayer()
    if (n==None):  return    
    s=getSelection()
    if (s==None):   return
    data=n.pixelData(s.x(),s.y(),s.width(),s.height())
    image=QImage(data.data(),s.width(),s.height(),QImage.Format_RGBA8888).rgbSwapped()

    # image = image.scaled(512,512, Qt.KeepAspectRatio, Qt.SmoothTransformation)        # not using config here
    print(image.width(),image.height())        
    data = QByteArray()
    buf = QBuffer(data)
    image.save(buf, 'PNG')
    ba=data.toBase64()
    DataAsString=str(ba,"ascii")
    #image64 = "data:image/png;base64,"+DataAsString
    image64 = DataAsString

    maskImage=QPixmap(image.width(), image.height()).toImage()
    maskImage = maskImage.convertToFormat(QImage.Format_ARGB32)
    # generate mask image
    maskImage.fill(QColor(Qt.black).rgb())
    foundTrans=False
    foundPixel=False
    for i in range(image.width()):
        for j in range(image.height()):
            rgb = image.pixel(i, j)
            alpha = qAlpha(rgb)
            if (alpha !=255):
                foundTrans=True
                maskImage.setPixel(i, j, QColor(Qt.white).rgb())
            else: foundPixel=True

    if (foundTrans==False):
        errorMessage("No transparent pixels found","Needs content with part removed by eraser (Brush in Tool palette + Right click for Eraser selection)")
        return
    if (foundPixel==False):
        errorMessage("No  pixels found","Maybe wrong layer selected? Choose one with some content n it.")
        return        
    data = QByteArray()
    buf = QBuffer(data)
    maskImage.save(buf, 'PNG')
    ba=data.toBase64()
    DataAsString=str(ba,"ascii")
   # maskImage64 = "data:image/png;base64,"+DataAsString
    maskImage64 =DataAsString
    SDConfig.load(SDConfig)
    image = image.scaled(380,380, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # preview smaller
    dlg = SDDialog("inpainting",image)
    dlg.resize(900,200)

    if dlg.exec():
        dlg.setDlgData()
        p = SDParameters()
        p.prompt=getFullPrompt(dlg)
        if not p.prompt: return        
        data=SDConfig.dlgData
        p.action="inpaint"
        p.negprompt = data["negprompt"]
        p.steps=data["steps"]
        p.seed=data["seed"]
        p.num=data["num"]
        p.scale=data["scale"]
        p.strength=data["strength"]
        p.scheduler=data["scheduler"]
        p.image64=image64
        p.maskImage64=maskImage64
        runSD(p)

# config dialog
def Config():
    dlg=SDConfigDialog()
    if dlg.exec():
        dlg.save()

# expand selection to max size        
def expandSelection():
    d = getDocument()

    if (d==None): return
    s = d.selection()    
    if (not s):  x=0;y=0
    else: x=s.x(); y=s.y()     
    s2 = Selection()    
    s2.select(x, y, SDConfig.width, SDConfig.height, 1)
    d.setSelection(s2)
    d.refreshProjection()


#Inpainting()
#TxtToImage()
#ImageToImage()
#Config()
#expandSelection()

