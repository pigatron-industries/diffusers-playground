from PIL import Image
import math, random
from ..ImageUtils import compositeImages, tiledImageProcessor
from .DiffusersPipelines import MAX_SEED, DiffusersPipelines
from huggingface_hub import login
from functools import partial

from IPython.display import display


def loginHuggingFace(token):
    login(token=token)


def tiledImageToImage(pipelines:DiffusersPipelines, initimg, prompt, negprompt, strength, scale, scheduler=None, seed=None, tilewidth=640, tileheight=640, overlap=128, model=None, callback=None):
    if(seed is None):
        seed = random.randint(0, MAX_SEED)
    
    def imageToImage(initimage):
        image, _ = pipelines.imageToImage(initimage=initimage, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, scheduler=scheduler, seed=seed, model=model)
        return image
    
    return tiledImageProcessor(imageToImage, initimg, tilewidth, tileheight, overlap, callback)


def tiledImageToImageOffset(pipelines:DiffusersPipelines, initimg, prompt, negprompt, strength, scale, scheduler=None, seed=None, 
                            tilewidth=640, tileheight=640, overlap=128, offsetx=0, offsety=0, model=None, callback=None):
    # creates a new image slightly bigger than original image to allow tiling to start at negative offset
    offsetimage = Image.new(initimg.mode, (initimg.width-offsetx, initimg.height-offsety))
    offsetimage.paste(initimg, (-offsetx, -offsety, -offsetx+initimg.width, -offsety+initimg.height))
    outimage, seed = tiledImageToImage(pipelines=pipelines, initimg=offsetimage, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, 
                                       scheduler=scheduler, seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, model=model, callback=callback)
    image = outimage.crop((-offsetx, -offsety, outimage.width, outimage.height))
    return image, seed


def tiledImageToImageCentred(pipelines:DiffusersPipelines, initimg, prompt, negprompt, strength, scale, scheduler=None, seed=None, 
                            tilewidth=640, tileheight=640, overlap=128, alignmentx='tile_centre', alignmenty='tile_centre', offsetx=0, offsety=0, model=None, callback=None):
    # find top left of initial centre tile 
    offsetx = offsetx + int(initimg.width/2)
    offsety = offsety + int(initimg.height/2)
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

    return tiledImageToImageOffset(pipelines, initimg=initimg, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, scheduler=scheduler, seed=seed, 
                            tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, offsetx=offsetx, offsety=offsety, model=model, callback=callback)


def compositedInpaint(pipelines:DiffusersPipelines, initimage, maskimage, prompt, negprompt, scale, steps=50, scheduler=None, seed=None, maskDilation=21, maskFeather=3, model=None):
    """ Standard inpaint but the result is composited back to the original using a feathered mask """
    outimage, usedseed = pipelines.inpaint(initimage=initimage, maskimage=maskimage, prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, scheduler=scheduler, seed=seed, model=model)
    outimage = compositeImages(outimage, initimage, maskimage, maskDilation=maskDilation, maskFeather=maskFeather)
    return outimage, usedseed


def tiledInpaint(pipelines, initimg, maskimg, prompt, negprompt, scale, steps=50, scheduler=None, seed=None, tilewidth=512, tileheight=512, overlap=128):
    if(seed is None):
        seed = random.randint(0, MAX_SEED)
    
    xslices = math.ceil((initimg.width) / (tilewidth-overlap))
    yslices = math.ceil((initimg.height) / (tileheight-overlap))
    print(f'Processing {xslices} x {yslices} slices')
    merged_image = initimg.convert("RGBA")

    # split into slices
    for yslice in range(yslices):
        for xslice in range(xslices):
            x = (xslice * (tilewidth - overlap))
            y = (yslice * (tileheight - overlap))
            image_slice = merged_image.crop((x, y, x+tilewidth, y+tileheight))
            mask_slice = maskimg.crop((x, y, x+tilewidth, y+tilewidth))

            image_slice = image_slice.convert("RGB")
            mask_slice = mask_slice.convert("RGB")

            display(image_slice)
            display(mask_slice)

            # TODO check if mask has any white in this slice, if all black then no inpainting necessary
            imageout_slice, _ = compositedInpaint(pipelines, initimage=image_slice, maskimage=mask_slice, prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, seed=seed, scheduler=scheduler)

            display(imageout_slice)
            
            # TODO merge original image in positions where there is no mask, to reduce seam
            imageout_slice = imageout_slice.convert("RGBA")
            merged_image.alpha_composite(imageout_slice, (x, y))

            # remove used mask
            maskimg.paste((0, 0, 0), [x, y, x+tilewidth, y+tileheight])

    return merged_image, seed


def tiledImageToImageInpaintSeams(pipelines, initimg, prompt, negprompt, strength, scale, scheduler=None, seed=None, tilewidth=512, tileheight=512, overlap=-64, model=None):
    # use negative overlap to leave gaps between tiles
    tiledImageToImageOffset(pipelines, initimg=initimg, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, scheduler=scheduler, seed=seed, 
                            tilewidth=tilewidth, tileheight=tileheight, overlap=-overlap, offsetx=0, offsety=0, model=model)
    pass


def tiledImageToImageMultipass(pipelines, initimg, prompt, negprompt, strength, scale, scheduler=None, seed=None, 
                               tilewidth=640, tileheight=640, overlap=128, passes=2, strengthMult=0.5, model=None, callback=None):
    offsetEven = (0, 0)
    offsetOdd = (-int((tilewidth - overlap)/2), -int((tileheight - overlap)/2))
    image = initimg

    for i in range(0, passes):
        if (i%2==0):
            offset = offsetEven
        else:
            offset = offsetOdd
        image, usedseed = tiledImageToImageOffset(pipelines, initimg=image, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, scheduler=scheduler, 
                                                  seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, offsetx=offset[0], offsety=offset[1], model=model, callback=callback)
        strength = strength * strengthMult

    return image, usedseed
