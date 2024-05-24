from PIL import Image, ImageDraw, ImageFilter, ImageOps
from io import BytesIO
from typing import List
from skimage import exposure
from blendmodes.blend import blendLayers, BlendType
import numpy as np
import base64
import cv2
import math

from IPython.display import display

def base64EncodeImage(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64image = base64.b64encode(buffered.getvalue())
    return b64image.decode()


def base64DecodeImage(b64image):
    buffer = BytesIO(base64.b64decode(b64image))
    image = Image.open(buffer)
    return image

def base64DecodeImages(b64images:List[Image.Image]) -> List[Image.Image]:
    if (b64images is None):
        return []
    images = []
    for b64image in b64images:
        images.append(base64DecodeImage(b64image))
    return images


def alphaToMask(image, smooth=False):
    maskimage = Image.new(image.mode, (image.width, image.height))
    maskimage.paste((0, 0, 0), [0, 0, image.width, image.height])
    for x in range(image.width):
        for y in range(image.height):
            pixel = image.getpixel((x, y))
            a = pixel[3]
            if (smooth):
                maskimage.putpixel((x, y), (255-a, 255-a, 255-a))
            else:
                if (a < 255):
                    maskimage.putpixel((x, y), (255, 255, 255))
    return maskimage


def invertAlpha(image, target):
    img = image.convert("RGBA")
    target_img = target.convert("RGBA")
    for x in range(img.width):
        for y in range(img.height):
            r, g, b, a = img.getpixel((x, y))
            r_t, g_t, b_t, a_t = target_img.getpixel((x, y))
            target_img.putpixel((x, y), (r_t, g_t, b_t, 255 - a))
            img.putpixel((x, y), (r, g, b, 255))
    return img, target_img


def removeAlpha(image):
    rgb_image = Image.new("RGB", image.size, (0, 0, 0))
    for x in range(image.width):
        for y in range(image.height):
            r, g, b, a = image.getpixel((x, y))
            rgb_image.putpixel((x, y), (r, g, b))
    return rgb_image


def compositeImages(foreground:Image.Image, background:Image.Image, mask:Image.Image, maskDilation:int=21, maskFeather:int=3):
    foreground = foreground.convert("RGBA")
    background = background.convert("RGBA")
    mask = mask.convert("L")
    dilated_mask = mask.filter(ImageFilter.MaxFilter(maskDilation))
    feathered_mask = dilated_mask.filter(ImageFilter.GaussianBlur(radius=maskFeather))
    return Image.composite(foreground, background, feathered_mask)


# create alpha mask with gradient at border
def createMask(width:int, height:int, border:int, top=False, bottom=False, left=False, right=False):
    mask = Image.new('L', (width, height), color=0xFF)
    draw = ImageDraw.Draw(mask)
    a = 0
    i = 0
    shape = ((0,0), (width, height))
    while i < border:
        draw.rectangle(shape, fill = a)
        i += 1
        a = int(256/border)*i
        x1 = 0 if left else i
        y1 = 0 if top else i
        x2 = width if right else width-i
        y2 = height if bottom else height-i
        shape = ((x1, y1), (x2, y2))
    return mask


# convert mask to alpha channel
def createAlphaMask(width:int, height:int, border:int, top=False, bottom=False, left=False, right=False):
    mask = createMask(width, height, border, top, bottom, left, right)
    alpha = Image.new('RGBA', (width, height), color=0)
    alpha.putalpha(mask)
    return alpha


def applyColourCorrection(fromimage, toimage):
    target = cv2.cvtColor(np.asarray(toimage.copy()), cv2.COLOR_RGB2LAB)
    source = cv2.cvtColor(np.asarray(fromimage.copy()), cv2.COLOR_RGB2LAB)
    matched = exposure.match_histograms(target, source, channel_axis=2)
    outimage = Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_LAB2RGB).astype("uint8"))
    outimage = blendLayers(outimage, toimage, BlendType.LUMINOSITY)
    return outimage


def pilToCv2(pil_image):
    return np.array(pil_image.convert("RGBA"))


def cv2ToPil(cv2_image):
    return Image.fromarray(cv2_image, "RGBA")


def tiledImageProcessor(processor, initimage, controlimages=None, tilewidth=640, tileheight=640, overlap=128, reduceEdges = False, scale=1, callback=None):
    xslices = math.ceil((initimage.width-overlap) / (tilewidth-overlap))
    yslices = math.ceil((initimage.height-overlap) / (tileheight-overlap))
    totalslices = xslices * yslices
    slicesdone = 0
    print(f'Processing {xslices} x {yslices} slices')
    if(callback is not None):
        callback("Running", totalslices, slicesdone)

    if(overlap >= 0):
        merged_image = initimage.convert("RGBA")
        merged_image = merged_image.resize((initimage.width*scale, initimage.height*scale), resample=Image.BICUBIC)
    else:
        # if overlap is negative create new transparent image to leave gaps between tiles
        merged_image = Image.new("RGBA", size=(initimage.width*scale, initimage.height*scale), color=(255, 255, 255, 0))

    # split into slices
    for yslice in range(yslices):
        for xslice in range(xslices):
            top = (yslice == 0)
            bottom = (yslice == yslices-1)
            left = (xslice == 0)
            right = (xslice == xslices-1)
            xleft = (xslice * (tilewidth - overlap))
            ytop = (yslice * (tileheight - overlap))
            xright = xleft + tilewidth
            ybottom = ytop + tileheight
            if(reduceEdges):
                if(bottom):
                    ybottom = initimage.height
                if(right):
                    xright = initimage.width

            if(overlap >= 0 and scale == 1): 
                # if possible take slice from merged image to include overlapped portions
                image_slice = merged_image.crop((xleft, ytop, xright, ybottom))
            else:
                image_slice = initimage.crop((xleft, ytop, xright, ybottom))
            
            # slice controlimages if provided
            controlimage_slices = None
            if(controlimages is not None):
                controlimage_slices = []
                for controlimage in controlimages:
                    controlimage_slice = controlimage.crop((xleft, ytop, xright, ybottom))
                    controlimage_slices.append(controlimage_slice)

            # process image tile
            image_slice = image_slice.convert("RGB")
            if(controlimages is not None):
                imageout_slice = processor(image_slice, controlimage_slices)
            else:
                imageout_slice = processor(image_slice)
            display(imageout_slice)
            # imageout_slice = applyColourCorrection(image_slice, imageout_slice)
            
            # merge image tile back into output image
            if(overlap >= 0):
                mask = createAlphaMask(tilewidth*scale, tileheight*scale, overlap//2, top, bottom, left, right)
                imageout_slice = imageout_slice.convert("RGBA")
                if(imageout_slice.width != mask.width or imageout_slice.height != mask.height):
                    imageout_slice = ImageOps.expand(imageout_slice, border=(0, 0, mask.width-imageout_slice.width, mask.height-imageout_slice.height), fill=(0, 0, 0, 0))
                imr, img, imb, _ = imageout_slice.split()
                mmr, mmg, mmb, mma = mask.split()
                finished_slice = Image.merge('RGBA', [imr, img, imb, mma])  # we want the RGB from the original, but the transparency from the mask
            else:
                finished_slice = imageout_slice.convert("RGBA")

            merged_image.alpha_composite(finished_slice, (xleft*scale, ytop*scale))

            if(callback is not None):
                slicesdone = slicesdone + 1
                callback("Running", totalslices, slicesdone, finished_slice)

    return merged_image