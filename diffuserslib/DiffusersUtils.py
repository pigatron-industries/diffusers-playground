from PIL import Image
import math, random
from .ImageUtils import createMask
from .DiffusersPipelines import MAX_SEED
from huggingface_hub import login


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


def tiledImageToImageOffset(pipelines, initimg, prompt, negprompt, strength, scale,  scheduler=None, seed=None, 
                            tilewidth=640, tileheight=640, overlap=128, offsetx=0, offsety=0):
    offsetimage = Image.new(initimg.mode, (initimg.width+offsetx, initimg.height+offsety))
    offsetimage.paste(initimg, (offsetx, offsety, offsetx+initimg.width, offsety+initimg.height))
    outimage, seed = tiledImageToImage(pipelines, offsetimage, prompt, negprompt, strength/2, scale, scheduler, seed, tilewidth, tileheight, overlap)
    outimage.crop((offsetx, offsety, outimage.width, outimage.height))
    return outimage, seed


def tiledImageToImageMultipass(pipelines, initimg, prompt, negprompt, strength, scale,  scheduler=None, seed=None, tilewidth=640, tileheight=640, overlap=128):
    image, seed = tiledImageToImage(pipelines, initimg, prompt, negprompt, strength, scale, scheduler, seed, tilewidth, tileheight, overlap)
    offsetx = int((tilewidth - overlap)/2)
    offsety = int((tileheight - overlap)/2)
    image, seed = tiledImageToImageOffset(pipelines, image, prompt, negprompt, strength, scale, scheduler, seed, tilewidth, tileheight, overlap, offsetx, offsety)
    return image, seed
