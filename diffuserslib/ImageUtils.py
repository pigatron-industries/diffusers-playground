from PIL import Image, ImageDraw

def base64EncodeImage(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64image = base64.b64encode(buffered.getvalue())
    return b64image.decode()


def base64DecodeImage(b64image):
    buffer = BytesIO(base64.b64decode(b64image))
    image = Image.open(buffer)
    return image

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