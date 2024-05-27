from imgcat import imgcat
from PIL import Image

image = Image.open('pose.png')

imgcat(image)