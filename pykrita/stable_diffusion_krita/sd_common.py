from krita import *
from PyQt5.Qt import QByteArray, QBuffer, QImage

def errorMessage(text,detailed):
    msgBox= QMessageBox()
    msgBox.resize(500,200)
    msgBox.setWindowTitle("Stable Diffusion")
    msgBox.setText(text)
    msgBox.setDetailedText(detailed)
    msgBox.setStyleSheet("QLabel{min-width: 700px;}")
    msgBox.exec()


def getDocument():
    d = Application.activeDocument()
    if (d==None):  
        errorMessage("Please add a document", "Needs document with a layer and selection.")
    return d


def getLayer():
    d = getDocument()
    if (d==None):  
        return
    n = d.activeNode()
    print(n.type())
    if(n.type() == "paintlayer"):
        return [n]
    elif(n.type() == "grouplayer"):
        return n.childNodes()
    errorMessage("Select a paint layer or group layer",  "Selected layer must be a paint layer or group layer.")
    return


def getSelection():
    d = getDocument()
    if (d==None): 
        return
    s = d.selection()
    if (s==None):  
        errorMessage("Please make a selection", "Operation runs on a selection only. Please use rectangle select tool.")
    return s   


def getLayerSelection():
    doc = getDocument()
    layers = getLayer()
    if (layers==None or len(layers)==0):
        return  
    layer = layers[0]
    selection = doc.selection()
    if(selection is None):
        data = layer.pixelData(0, 0, doc.width(), doc.height())
        image = QImage(data.data(),doc.width(),doc.height(),QImage.Format_RGBA8888).rgbSwapped()
    else:
        data=layer.pixelData(selection.x(), selection.y(), selection.width(), selection.height())
        image = QImage(data.data(), selection.width(), selection.height(), QImage.Format_RGBA8888).rgbSwapped()
    return image


def base64EncodeImage(image):
    data = QByteArray()
    buf = QBuffer(data)
    image.save(buf, 'PNG')
    ba=data.toBase64()
    image64=str(ba,"ascii")
    return image64