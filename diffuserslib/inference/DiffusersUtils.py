from PIL import Image
import math, random
from ..ImageUtils import createMask, compositeImages, applyColourCorrection
from .DiffusersPipelines import MAX_SEED
from huggingface_hub import login

from IPython.display import display


def loginHuggingFace(token):
    login(token=token)


def tiledImageToImage(pipelines, initimg, prompt, negprompt, strength, scale, scheduler=None, seed=None, tilewidth=640, tileheight=640, overlap=128):
    if(seed is None):
        seed = random.randint(0, MAX_SEED)
    
    xslices = math.ceil((initimg.width) / (tilewidth-overlap))
    yslices = math.ceil((initimg.height) / (tileheight-overlap))
    print(f'Processing {xslices} x {yslices} slices')
    if(overlap >= 0):
        merged_image = initimg.convert("RGBA")
    else:
        # if overlap is negative create new transparent image to leave gaps between tiles
        merged_image = Image.new("RGBA", size=initimg.size, color=(255, 255, 255, 0))

    # split into slices
    for yslice in range(yslices):
        for xslice in range(xslices):
            top = (yslice == 0)
            bottom = (yslice == yslices-1)
            left = (xslice == 0)
            right = (xslice == xslices-1)
            x = (xslice * (tilewidth - overlap))
            y = (yslice * (tileheight - overlap))
            
            if(overlap >= 0):
                image_slice = merged_image.crop((x, y, x+tilewidth, y+tileheight))
            else:
                image_slice = initimg.crop((x, y, x+tilewidth, y+tileheight))

            image_slice = image_slice.convert("RGB")
            imageout_slice, _ = pipelines.imageToImage(image_slice, prompt, negprompt, strength, scale, seed, scheduler)
            imageout_slice = applyColourCorrection(image_slice, imageout_slice)
            
            if(overlap >= 0):
                mask = createMask(tilewidth, tileheight, overlap/2, top, bottom, left, right)
                imr, img, imb, _ = imageout_slice.split()
                mmr, mmg, mmb, mma = mask.split()
                finished_slice = Image.merge('RGBA', [imr, img, imb, mma])  # we want the RGB from the original, but the transparency from the mask
            else:
                finished_slice = imageout_slice.convert("RGBA")

            merged_image.alpha_composite(finished_slice, (x, y))

    return merged_image, seed


def tiledImageToImageOffset(pipelines, initimg, prompt, negprompt, strength, scale, scheduler=None, seed=None, 
                            tilewidth=640, tileheight=640, overlap=128, offsetx=0, offsety=0):
    # creates a new image slightly bigger than original image to allow tiling to start at negative offset
    offsetimage = Image.new(initimg.mode, (initimg.width-offsetx, initimg.height-offsety))
    offsetimage.paste(initimg, (-offsetx, -offsety, -offsetx+initimg.width, -offsety+initimg.height))
    outimage, seed = tiledImageToImage(pipelines, offsetimage, prompt, negprompt, strength, scale, scheduler, seed, tilewidth, tileheight, overlap)
    image = outimage.crop((-offsetx, -offsety, outimage.width, outimage.height))
    return image, seed


def tiledImageToImageCentred(pipelines, initimg, prompt, negprompt, strength, scale, scheduler=None, seed=None, 
                            tilewidth=640, tileheight=640, overlap=128, alignmentx='tile_centre', alignmenty='tile_centre', offsetx=0, offsety=0):
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
                            tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, offsetx=offsetx, offsety=offsety)


def compositedInpaint(pipelines, initimage, maskimage, prompt, negprompt, scale, steps=50, scheduler=None, seed=None, maskDilation=21, maskFeather=3):
    """ Standard inpaint but the result is composited back to the original using a feathered mask """
    outimage, usedseed = pipelines.inpaint(initimage=initimage, maskimage=maskimage, prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, scheduler=scheduler, seed=seed)
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


def tiledImageToImageInpaintSeams(pipelines, initimg, prompt, negprompt, strength, scale, scheduler=None, seed=None, tilewidth=512, tileheight=512, overlap=-64):
    # use negative overlap to leave gaps between tiles
    tiledImageToImageOffset(pipelines, initimg=initimg, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, scheduler=scheduler, seed=seed, 
                            tilewidth=tilewidth, tileheight=tileheight, overlap=-overlap, offsetx=0, offsety=0)
    pass


def tiledImageToImageMultipass(pipelines, initimg, prompt, negprompt, strength, scale, scheduler=None, seed=None, 
                               tilewidth=640, tileheight=640, overlap=128, passes=2, strengthMult=0.5):
    offsetEven = (0, 0)
    offsetOdd = (-int((tilewidth - overlap)/2), -int((tileheight - overlap)/2))
    image = initimg

    for i in range(0, passes):
        if (i%2==0):
            offset = offsetEven
        else:
            offset = offsetOdd
        image, usedseed = tiledImageToImageOffset(pipelines, initimg=image, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, scheduler=scheduler, 
                                                  seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, offsetx=offset[0], offsety=offset[1])
        strength = strength * strengthMult

    return image, usedseed
