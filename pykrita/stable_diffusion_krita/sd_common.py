from krita import Krita, Document, Node, QMessageBox # type: ignore
from PyQt5.Qt import QByteArray, QBuffer, QImage # type: ignore
from typing import List

def errorMessage(text,detailed):
    msgBox= QMessageBox()
    msgBox.resize(500,200)
    msgBox.setWindowTitle("Stable Diffusion")
    msgBox.setText(text)
    msgBox.setDetailedText(detailed)
    msgBox.setStyleSheet("QLabel{min-width: 700px;}")
    msgBox.exec()


def getDocument() -> "Document|None":
    doc = Krita.instance().activeDocument()
    if (doc == None):  
        errorMessage("Please add a document", "Needs document with a layer and selection.")
    return doc


def getLayer() -> "Node|None":
    doc = getDocument()
    if (doc==None):  
        return
    node = doc.activeNode()
    print(node.type())
    if(node.type() == "paintlayer"):
        return [node]
    elif(node.type() == "grouplayer"):
        return node.childNodes()
    errorMessage("Select a paint layer or group layer",  "Selected layer must be a paint layer or group layer.")
    return


def getSelection():
    doc = getDocument()
    if (doc==None): 
        return
    selection = doc.selection()
    if (selection==None):  
        return doc
        # errorMessage("Please make a selection", "Operation runs on a selection only. Please use rectangle select tool.")
    return selection


def getLayerSelections() -> List[QImage]:
    doc = getDocument()
    if (doc == None):
        return []
    layers = getLayer()
    if (layers==None or len(layers)==0):
        return []
    selection = doc.selection()
    images = []
    for layer in layers:
        if(selection is None):
            data = layer.pixelData(0, 0, doc.width(), doc.height())
            images.append(QImage(data.data(),doc.width(),doc.height(),QImage.Format_RGBA8888).rgbSwapped())
        else:
            data=layer.pixelData(selection.x(), selection.y(), selection.width(), selection.height())
            images.append(QImage(data.data(), selection.width(), selection.height(), QImage.Format_RGBA8888).rgbSwapped())
    return images


def base64EncodeImage(image):
    data = QByteArray()
    buf = QBuffer(data)
    image.save(buf, 'PNG')
    ba=data.toBase64()
    image64=str(ba,"ascii")
    return image64


def base64EncodeImages(images):
    images64 = []
    for image in images:
        images64.append(base64EncodeImage(image))
    return images64