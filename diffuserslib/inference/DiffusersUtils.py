from PIL import Image
import random, copy
from ..ImageUtils import compositeImages, tiledImageProcessor, applyColourCorrection
from .DiffusersPipelines import MAX_SEED, DiffusersPipelines
from .GenerationParameters import ControlImageParameters
from .GenerationParameters import GenerationParameters, ControlImageType
from huggingface_hub import login
from typing import List


def loginHuggingFace(token):
    login(token=token)


def tiledGeneration(pipelines:DiffusersPipelines, params:GenerationParameters, tilewidth=768, tileheight=768, overlap=128, masktile:ControlImageParameters|None=None, callback=None):
    initimageparams = params.getInitImage()
    if initimageparams is not None:
        initimage = initimageparams.image
    else:
        initimage = None
    
    controlimages = [ controlimageparams.image for controlimageparams in params.getImages(ControlImageType.IMAGETYPE_CONTROLIMAGE) ]
    if(params.seed is None):
        params.seed = random.randint(0, MAX_SEED)
    
    def imageToImageFunc(initimagetile:Image.Image|None, controlimagetiles:List[Image.Image]):
        tileparams = copy.deepcopy(params)
        if(initimagetile is not None):
            tileparams.width = initimagetile.width
            tileparams.height = initimagetile.height
        elif(controlimagetiles is not None and len(controlimagetiles) > 0):
            tileparams.width = controlimagetiles[0].width
            tileparams.height = controlimagetiles[0].height
        tileparams.generationtype = "generate"
        if(initimagetile is not None):
            tileparams.setInitImage(initimagetile)
        if masktile is not None:
            tileparams.controlimages.append(masktile)
        for i in range(len(controlimagetiles)):
            tileparams.setControlImage(i, controlimagetiles[i])
        image, _ = pipelines.generate(tileparams)
        return image
    
    return tiledImageProcessor(processor=imageToImageFunc, initimage=initimage, controlimages=controlimages, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, callback=callback), params.seed


def tiledInpaint(pipelines:DiffusersPipelines, params:GenerationParameters, tilewidth=768, tileheight=768, overlap=256, inpaintwidth=512, inpaintheight=512, callback=None):
    # create centred rectangle mask image
    masktile = Image.new("RGB", size=(tilewidth, tileheight), color=(0, 0, 0))
    masktile.paste((255, 255, 255), (int((tilewidth/2)-(inpaintwidth/2)), int((tileheight/2)-(inpaintheight/2)), int((tilewidth/2)+(inpaintwidth/2)), int((tileheight/2)+(inpaintheight/2))))
    masktileparams = ControlImageParameters(image=masktile, type=ControlImageType.IMAGETYPE_MASKIMAGE)
    return tiledGeneration(pipelines=pipelines, params=params, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, masktile=masktileparams, callback=callback)
    

def tiledProcessorOffset(tileprocessor, pipelines:DiffusersPipelines, params:GenerationParameters, tilewidth:int=640, tileheight:int=640, overlap:int=128, offsetx:int=0, offsety:int=0):
    # creates a new image slightly bigger than original image to allow tiling to start at negative offset   
    params = copy.deepcopy(params)

    for controlimageparams in params.controlimages:
        offsetcontrolimage = Image.new(controlimageparams.image.mode, (controlimageparams.image.width-offsetx, controlimageparams.image.height-offsety))
        offsetcontrolimage.paste(controlimageparams.image, (-offsetx, -offsety, -offsetx+controlimageparams.image.width, -offsety+controlimageparams.image.height))
        controlimageparams.image = offsetcontrolimage

    outimage, seed = tileprocessor(pipelines=pipelines, params=params, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap)
    image = outimage.crop((-offsetx, -offsety, outimage.width, outimage.height))
    return image, seed


def tiledProcessorCentred(tileprocessor, pipelines:DiffusersPipelines, params:GenerationParameters, tilewidth=640, tileheight=640, overlap=128, 
                             alignmentx='tile_centre', alignmenty='tile_centre', offsetx=0, offsety=0):
    # find top left of initial centre tile 
    offsetx = offsetx + int(params.width/2)
    offsety = offsety + int(params.height/2)
    if(alignmentx == 'tile_centre'):
        offsetx = offsetx - int(tilewidth/2)
    else:
        offsetx = offsetx - int(tilewidth-(overlap/2))
    if(alignmenty == 'tile_centre'):
        offsety = offsety - int(tileheight/2)
    else:
        offsety = offsety - int(tileheight-(overlap/2))

    # find top left of first tile outside of image
    while offsetx > 0:
        offsetx = offsetx - (tilewidth-overlap)
    while offsety > 0:
        offsety = offsety - (tileheight-overlap)

    return tiledProcessorOffset(tileprocessor, pipelines=pipelines, params=params, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, offsetx=offsetx, offsety=offsety)


def compositedInpaint(pipelines:DiffusersPipelines, params:GenerationParameters, maskDilation=21, maskFeather=3):
    """ Standard inpaint but the result is composited back to the original using a feathered mask """
    outimage, usedseed = pipelines.generate(params)
    initimageparams = params.getInitImage()
    maskimageparams = params.getMaskImage()
    if initimageparams is None or maskimageparams is None:
        raise Exception("compositedInpaint requires initimage and maskimage to be set in params")
    # outimage = applyColourCorrection(initimageparams.image, outimage)
    outimage = compositeImages(outimage, initimageparams.image, maskimageparams.image, maskDilation=maskDilation, maskFeather=maskFeather)
    return outimage, usedseed


def tiledImageToImageMultipass(tileprocessor, pipelines:DiffusersPipelines, params:GenerationParameters, tilewidth=640, tileheight=640, overlap=128, passes=2, strength=0.2, strengthMult=0.5):
    offsetEven = (0, 0)
    offsetOdd = (-int((tilewidth - overlap)/2), -int((tileheight - overlap)/2))
    image = None
    usedseed = None

    for i in range(0, passes):
        if (i%2==0):
            offset = offsetEven
        else:
            offset = offsetOdd
        image, usedseed = tiledProcessorOffset(tileprocessor=tileprocessor, pipelines=pipelines, params=params, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, 
                                               offsetx=offset[0], offsety=offset[1])
        params = copy.deepcopy(params)
        params.strength = strength * strengthMult

    return image, usedseed
