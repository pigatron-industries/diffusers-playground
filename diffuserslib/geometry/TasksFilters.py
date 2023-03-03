from .GeometryPipeline import GeometryTask
from .. import evaluateArguments

from PIL import ImageEnhance


class SaturationTask(GeometryTask):
    def __init__(self, saturation = 0):
        self.args = {
            "saturation": saturation
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        converter = ImageEnhance.Color(context.image)
        context.image = converter.enhance(args["saturation"]+1)
        return context