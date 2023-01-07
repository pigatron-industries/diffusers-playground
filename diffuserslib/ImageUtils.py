from PIL import Image, ImageDraw
from io import BytesIO
import base64

def base64EncodeImage(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64image = base64.b64encode(buffered.getvalue())
    return b64image.decode()


def base64DecodeImage(b64image):
    buffer = BytesIO(base64.b64decode(b64image))
    image = Image.open(buffer)
    return image


def alphaToMask(image):
    maskimage = Image.new(image.mode, (image.width, image.height))
    maskimage.paste((0, 0, 0), [0, 0, image.width, image.height])
    for x in range(image.width):
        for y in range(image.height):
            pixel = image.getpixel((x, y))
            if (pixel[3] < 255):
                maskimage.putpixel((x, y), (255, 255, 255))
    return maskimage;


def compositeImages(foreground_image, background_image):
    # alpha channel must be on background and is inverted
    foreground = foreground_image.convert("RGBA")
    background = background_image.convert("RGBA")
    composite = Image.new("RGBA", background.size, (0, 0, 0, 0))
    composite.paste(foreground, (0, 0), background)
    background.paste(composite, (0, 0), composite)
    return background


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