from PIL import Image
import math
from ImageUtils import createMask


def tiledImageToImage(pipelines, initimg, prompt, negprompt, strength, scale, seed=None, tilewidth=640, tileheight=640, overlap=128):
    xslices = math.ceil(initimg.width / (tilewidth-overlap))
    yslices = math.ceil(initimg.height / (tileheight-overlap))
    print(f'Processing {xslices} x {yslices} slices')
    merged_image = initimg.convert("RGBA")

    # split into slices
    for yslice in range(yslices):
        for xslice in range(xslices):
            top = (yslice == 0)
            bottom = (yslice == yslices-1)
            left = (xslice == 0)
            right = (xslice == xslices-1)
            mask = createMask(tilewidth, tileheight, overlap, top, bottom, left, right)
            
            x = xslice * (tilewidth - overlap)
            y = yslice * (tileheight - overlap)
            image_slice = merged_image.crop((x, y, x+tilewidth, y+tileheight))

            image_slice = image_slice.convert("RGB")
            imageout_slice, _ = pipelines.imageToImage(image_slice, prompt, negprompt, strength, scale, seed)
            
            imr, img, imb = imageout_slice.split()
            mmr, mmg, mmb, mma = mask.split()
            finished_slice = Image.merge('RGBA', [imr, img, imb, mma])  # we want the RGB from the original, but the transparency from the mask
            merged_image.alpha_composite(finished_slice, (x, y))

    return merged_image