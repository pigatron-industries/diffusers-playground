from PIL import Image
import math, random
from .ImageUtils import createMask, compositeImages
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
    merged_image = initimg.convert("RGBA")

    # split into slices
    for yslice in range(yslices):
        for xslice in range(xslices):
            top = (yslice == 0)
            bottom = (yslice == yslices-1)
            left = (xslice == 0)
            right = (xslice == xslices-1)
            mask = createMask(tilewidth, tileheight, overlap/2, top, bottom, left, right)
            
            x = (xslice * (tilewidth - overlap))
            y = (yslice * (tileheight - overlap))
            image_slice = merged_image.crop((x, y, x+tilewidth, y+tileheight))

            image_slice = image_slice.convert("RGB")
            imageout_slice, _ = pipelines.imageToImage(image_slice, prompt, negprompt, strength, scale, seed, scheduler)
            
            imr, img, imb = imageout_slice.split()
            mmr, mmg, mmb, mma = mask.split()
            finished_slice = Image.merge('RGBA', [imr, img, imb, mma])  # we want the RGB from the original, but the transparency from the mask
            merged_image.alpha_composite(finished_slice, (x, y))

    return merged_image, seed


def compositedInpaint(pipelines, initimage, maskimage, prompt, negprompt, scale, steps=50, scheduler=None, seed=None, maskDilation=21, maskFeather=3):
    """ Standard inpaint but the result is composited back to the original using a feathered mask """
    outimage, usedseed = pipelines.inpaint(initmage=initimage, maskimage=maskimage, prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, scheduler=scheduler, seed=seed)
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
            imageout_slice, _ = compositedInpaint(pipelines, image_slice, mask_slice, prompt, negprompt, steps, scale, seed, scheduler)

            display(imageout_slice)
            
            # TODO merge original image in positions where there is no mask, to reduce seam
            imageout_slice = imageout_slice.convert("RGBA")
            merged_image.alpha_composite(imageout_slice, (x, y))

            # remove used mask
            maskimg.paste((0, 0, 0), [x, y, x+tilewidth, y+tileheight])

    return merged_image, seed


def tiledImageToImageInpaintSeams(pipelines, initimg, prompt, negprompt, strength, scale, scheduler=None, seed=None, tilewidth=640, tileheight=640, overlap=128):
    pass


def tiledImageToImageOffset(pipelines, initimg, prompt, negprompt, strength, scale,  scheduler=None, seed=None, 
                            tilewidth=640, tileheight=640, overlap=128, offsetx=0, offsety=0):
    offsetimage = Image.new(initimg.mode, (initimg.width+offsetx, initimg.height+offsety))
    offsetimage.paste(initimg, (offsetx, offsety, offsetx+initimg.width, offsety+initimg.height))
    outimage, seed = tiledImageToImage(pipelines, offsetimage, prompt, negprompt, strength/2, scale, scheduler, seed, tilewidth, tileheight, overlap)
    image = outimage.crop((offsetx, offsety, outimage.width, outimage.height))
    return image, seed


def tiledImageToImageMultipass(pipelines, initimg, prompt, negprompt, strength, scale, scheduler=None, seed=None, 
                               tilewidth=640, tileheight=640, overlap=128, passes=2, strengthMult=0.5):
    offsetEven = (0, 0)
    offsetOdd = (int((tilewidth - overlap)/2), int((tileheight - overlap)/2))
    image = initimg

    for i in range(0, passes):
        if (i%2==0):
            offset = offsetEven
        else:
            offset = offsetOdd
        strength = strength * strengthMult
        image, usedseed = tiledImageToImageOffset(pipelines, initimg=image, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, scheduler=scheduler, 
                                                  seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, offsetx=offset[0], offsety=offset[1])

    return image, usedseed
