from PIL import Image
import math, random
from ..ImageUtils import compositeImages, tiledImageProcessor
from .DiffusersPipelines import MAX_SEED, DiffusersPipelines
from huggingface_hub import login
from typing import List, Optional

from IPython.display import display


def loginHuggingFace(token):
    login(token=token)


def tiledImageToImage(pipelines:DiffusersPipelines, initimage, prompt, negprompt, strength, scale, scheduler=None, seed=None, 
                      controlimages=None, controlmodels=None, model=None, tilewidth=768, tileheight=768, overlap=128, callback=None):
    if(seed is None):
        seed = random.randint(0, MAX_SEED)
    
    def imageToImageFunc(initimagetile, controlimagetiles=None):
        if(controlimagetiles is None or len(controlimagetiles) == 0):
            image, _ = pipelines.imageToImage(initimage=initimagetile, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, 
                                              scheduler=scheduler, seed=seed, model=model)
        else:
            image, _ = pipelines.imageToImageControlNet(initimage=initimagetile, controlimage=controlimagetiles, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, 
                                                        scheduler=scheduler, seed=seed, model=model, controlmodel=controlmodels)
        return image
    
    return tiledImageProcessor(processor=imageToImageFunc, initimage=initimage, controlimages=controlimages, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, callback=callback), seed


def tiledInpaint(pipelines:DiffusersPipelines, initimage, prompt, negprompt, strength, scale, scheduler=None, seed=None, 
                      controlimages=None, controlmodels=None, model=None, tilewidth=768, tileheight=768, overlap=256, inpaintwidth=512, inpaintheight=512, callback=None):
    if(seed is None):
        seed = random.randint(0, MAX_SEED)

    # create mask image
    mask = Image.new("RGB", size=(tilewidth, tileheight), color=(0, 0, 0))
    mask.paste((255, 255, 255), (int((tilewidth/2)-(inpaintwidth/2)), int((tileheight/2)-(inpaintheight/2)), int((tilewidth/2)+(inpaintwidth/2)), int((tileheight/2)+(inpaintheight/2))))
    
    def inpaintFunc(initimagetile, controlimagetiles=None):
        image, _ = compositedInpaint(pipelines=pipelines, initimage=initimagetile, maskimage=mask, controlimage=controlimagetiles, prompt=prompt, negprompt=negprompt, 
                                     strength=strength, scale=scale, steps=50, scheduler=scheduler, seed=seed, model=model, controlmodel=controlmodels)
        return image
    
    return tiledImageProcessor(processor=inpaintFunc, initimage=initimage, controlimages=controlimages, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, callback=callback), seed



def tiledProcessorOffset(tileprocessor, initimage:Image.Image, controlimages:List[Image.Image]|None=None, 
                         tilewidth:int=640, tileheight:int=640, overlap:int=128, offsetx:int=0, offsety:int=0, **kwargs):
    # creates a new image slightly bigger than original image to allow tiling to start at negative offset
    offsetimage = Image.new(initimage.mode, (initimage.width-offsetx, initimage.height-offsety))
    offsetimage.paste(initimage, (-offsetx, -offsety, -offsetx+initimage.width, -offsety+initimage.height))
    offsetcontrolimages = None
    if controlimages is not None:
        offsetcontrolimages = []
        for controlimage in controlimages:
            offsetcontrolimage = Image.new(controlimage.mode, (controlimage.width-offsetx, controlimage.height-offsety))
            offsetcontrolimage.paste(controlimage, (-offsetx, -offsety, -offsetx+controlimage.width, -offsety+controlimage.height))
            offsetcontrolimages.append(offsetcontrolimage)
    outimage, seed = tileprocessor(initimage=offsetimage, controlimages=offsetcontrolimages, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, **kwargs)
    image = outimage.crop((-offsetx, -offsety, outimage.width, outimage.height))
    return image, seed


def tiledProcessorCentred(tileprocessor, initimage, controlimages=None, tilewidth=640, tileheight=640, overlap=128, 
                             alignmentx='tile_centre', alignmenty='tile_centre', offsetx=0, offsety=0, **kwargs):
    # find top left of initial centre tile 
    offsetx = offsetx + int(initimage.width/2)
    offsety = offsety + int(initimage.height/2)
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

    return tiledProcessorOffset(tileprocessor, initimage=initimage, controlimages=controlimages, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, offsetx=offsetx, offsety=offsety, **kwargs)


def compositedInpaint(pipelines:DiffusersPipelines, initimage, maskimage, prompt, negprompt, scale, steps=50, strength=1.0, scheduler=None, seed=None, maskDilation=21, maskFeather=3, model=None, controlmodel=None, controlimage=None):
    """ Standard inpaint but the result is composited back to the original using a feathered mask """
    if(controlmodel is None or len(controlmodel) == 0):
        outimage, usedseed = pipelines.inpaint(initimage=initimage, maskimage=maskimage, prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, strength=strength, scheduler=scheduler, seed=seed, model=model)
    else:
        outimage, usedseed = pipelines.inpaintControlNet(initimage=initimage, maskimage=maskimage, prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, scheduler=scheduler, seed=seed, model=model, 
                                                         controlmodel=controlmodel, controlimage=controlimage)
    outimage = compositeImages(outimage, initimage, maskimage, maskDilation=maskDilation, maskFeather=maskFeather)
    return outimage, usedseed


def tiledImageToImageMultipass(tileprocessor, initimage, tilewidth=640, tileheight=640, overlap=128, passes=2, strength=0.2, strengthMult=0.5, **kwargs):
    offsetEven = (0, 0)
    offsetOdd = (-int((tilewidth - overlap)/2), -int((tileheight - overlap)/2))
    image = initimage

    for i in range(0, passes):
        if (i%2==0):
            offset = offsetEven
        else:
            offset = offsetOdd
        image, usedseed = tiledProcessorOffset(tileprocessor=tileprocessor, initimage=image, stregnth=strength, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, 
                                               offsetx=offset[0], offsety=offset[1], **kwargs)
        strength = strength * strengthMult

    return image, usedseed
