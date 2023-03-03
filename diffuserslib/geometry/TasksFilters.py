from .GeometryPipeline import GeometryTask
from .. import evaluateArguments

from PIL import Image


class HueSaturationLightnessTask(GeometryTask):
    def __init__(self, hue = 0, saturation = 0, lightness = 0):
        self.args = {
            "hue": hue,
            "saturation": saturation,
            "lightness": lightness,
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)

        hsv_image = context.image.convert('HSV')
        hue_channel, saturation_channel, brightness_channel = hsv_image.split()
                
        hue_adjusted_channel = hue_channel.point(lambda i: i + int(args["hue"] * 255))
        saturation_adjusted_channel = saturation_channel.point(lambda i: i + int(args["saturation"] * 255))
        brightness_adjusted_channel = brightness_channel.point(lambda i: i + int(args["brightness"] * 255))

        modified_hsv_image = Image.merge('HSV', (hue_adjusted_channel, saturation_adjusted_channel, brightness_adjusted_channel))
        context.image = modified_hsv_image.convert('RGB')
        return context