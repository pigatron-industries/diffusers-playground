from PIL import Image, ImageDraw, ImageFilter
from io import BytesIO
import base64

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
    return maskimage;


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


def compositeImages(foreground, background, mask, maskDilation=21, maskFeather=3):
    foreground = foreground.convert("RGBA")
    background = background.convert("RGBA")
    mask = mask.convert("L")
    dilated_mask = mask.filter(ImageFilter.MaxFilter(maskDilation))
    feathered_mask = dilated_mask.filter(ImageFilter.GaussianBlur(radius=maskFeather))
    return Image.composite(foreground, background, feathered_mask)


# create alpha mask with gradient at overlap
def createMask(width, height, overlap, top=False, bottom=False, left=False, right=False):
    alpha = Image.new('L', (width, height), color=0xFF)
    alpha_gradient = ImageDraw.Draw(alpha)
    a = 0
    i = 0
    shape = ((width, height), (0,0))
    while i < overlap:
        alpha_gradient.rectangle(shape, fill = a)
        i += 1
        a = int(256/overlap)*i
        x1 = 0 if left else i
        y1 = 0 if top else i
        x2 = width if right else width-i
        y2 = height if bottom else height-i
        shape = ((x2, y2), (x1,y1))

    mask = Image.new('RGBA', (width, height), color=0)
    mask.putalpha(alpha)
    return mask