from ..ImageProcessor import ImageProcessor, ImageContext
from ....batch import evaluateArguments


class DrawJuliaSetProcessor(ImageProcessor):
    def __init__(self, c_real, c_imaginary, xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0, max_iter=255):
        self.args = {
            "c_real": c_real,
            "c_imaginary": c_imaginary,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "max_iter": max_iter
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)

        xstep = (args["xmax"] - args["xmin"]) / (context.image.width - 1)
        ystep = (args["ymax"] - args["ymin"]) / (context.image.height - 1)

        for y in range(context.image.width):
            for x in range(context.image.height):
                z = complex(args["xmin"] + x * xstep, args["ymin"] + y * ystep)
                for i in range(args["max_iter"]):
                    z = z*z + complex(args["c_real"], args["c_imaginary"])
                    if abs(z) > 2.0:
                        break
                if i == args["max_iter"]-1:
                    context.image.putpixel((x, y), (255, 255, 255))
                else:
                    context.image.putpixel((x, y), (0, 0, 0))

        return context