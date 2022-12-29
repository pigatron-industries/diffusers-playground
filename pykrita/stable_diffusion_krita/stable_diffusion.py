from PyQt5.QtWidgets import *
from krita import *
from . import sd_main
class SDDocker(DockWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stable Diffusion")
        mainWidget = QWidget(self)
        self.setWidget(mainWidget)
        mainWidget.setLayout(QVBoxLayout())

        btnTxt2Txt = QPushButton("Txt 2 Img", mainWidget)
        btnImg2Img = QPushButton("Img 2 Img", mainWidget)
        btnDepth2Img = QPushButton("Depth 2 Img", mainWidget)
        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(btnTxt2Txt) 
        h_layout1.addWidget(btnImg2Img)
        h_layout1.addWidget(btnDepth2Img)   
        mainWidget.layout().addLayout(h_layout1)

        btnUpscale = QPushButton("Upscale", mainWidget)
        btnInpaint = QPushButton("Inpaint", mainWidget)
        # btnFaceEnhance = QPushButton("Face Enhance", mainWidget)
        btnTiledImg2Img = QPushButton("Tiled Img 2 Img", mainWidget)
        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(btnUpscale) 
        h_layout2.addWidget(btnInpaint) 
        # h_layout2.addWidget(btnFaceEnhance) 
        h_layout2.addWidget(btnTiledImg2Img) 
        mainWidget.layout().addLayout(h_layout2)

        btnConfig = QPushButton("", mainWidget)
        btnConfig.setIcon(Krita.instance().icon('configure'))
        btnSelection = QPushButton("", mainWidget)
        btnSelection.setIcon(Krita.instance().icon('tool_rect_selection'))
        h_layout3 = QHBoxLayout()
        h_layout3.addWidget(btnConfig)        
        h_layout3.addWidget(btnSelection)        
        mainWidget.layout().addLayout(h_layout3)
        btnTxt2Txt.clicked.connect(sd_main.TxtToImage)
        btnImg2Img.clicked.connect(sd_main.ImageToImage)
        btnDepth2Img.clicked.connect(sd_main.DepthToImage)
        btnInpaint.clicked.connect(sd_main.Inpaint)
        btnUpscale.clicked.connect(sd_main.Upscale)

        # btnFaceEnhance.clicked.connect(sd_main.FaceEnhance)
        btnTiledImg2Img.clicked.connect(sd_main.TiledImageToImage)
        btnConfig.clicked.connect(sd_main.Config)
        btnSelection.clicked.connect(sd_main.expandSelection)


    def canvasChanged(self, canvas):
        pass

Krita.instance().addDockWidgetFactory(DockWidgetFactory("Stable Diffusion", DockWidgetFactoryBase.DockRight, SDDocker))
