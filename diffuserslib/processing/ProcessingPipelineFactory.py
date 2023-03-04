from .ImageProcessor import ImageProcessorPipeline, InitImageProcessor, FillBackgroundProcessor
from .TransformProcessors import SimpleTransformProcessor, SymmetrizeProcessor, SpiralizeProcessor
from .GeometryProcessors import DrawRegularShapeProcessor, DrawCheckerboardProcessor
from .. import RandomNumberArgument, RandomChoiceArgument, RandomPositionArgument

import math


def simpleTransform(image, transform=RandomChoiceArgument(["fliphorizontal", "flipvertical", "rotate90", "rotate180", "rotate270", "none"])):
    geometry = ImageProcessorPipeline()
    geometry.addTask(InitImageProcessor(image))
    geometry.addTask(SimpleTransformProcessor(type=transform))
    return geometry


def shapeGeometryPipeline(size=(512, 512), background="white", foreground="black", sides=RandomNumberArgument(3, 6), minsize=32, maxsize=256, shapes=1,
                          symmetry=RandomChoiceArgument(["horizontal", "vertical", "rotation", "none"])):
    geometry = ImageProcessorPipeline(size=size)
    for i in range(shapes):
        geometry.addTask(DrawRegularShapeProcessor(
            position=RandomPositionArgument(), 
            size=RandomNumberArgument(minsize, maxsize),
            sides=sides,
            fill=foreground
        ))
    geometry.addTask(SymmetrizeProcessor(type=symmetry))
    geometry.addTask(FillBackgroundProcessor(background = background))
    return geometry


def spiralGeometryPipeline(size=(512, 512), background="white", sides=RandomNumberArgument(3, 6), minsize=32, maxsize=256, shapes=1, rotation=360, steps=8, zoom=4):
    geometry = ImageProcessorPipeline(size=size)
    for i in range(shapes):
        geometry.addTask(DrawRegularShapeProcessor(
            position=RandomPositionArgument(), 
            size=RandomNumberArgument(minsize, maxsize),
            sides=sides
        ))
    geometry.addTask(SpiralizeProcessor(
        rotation = rotation, steps = steps, zoom = zoom
    ))
    geometry.addTask(FillBackgroundProcessor(background = background))
    return geometry


def checkerboardGeometryPipeline(size=(512, 512), background="white", foreground="black", blocksize = None, start = "black"):
    if(blocksize is None):
        blocksize = RandomChoiceArgument([
            #  aspect ratio dependent
            (math.ceil(size[0]/2), math.ceil(size[1]/2)), 
            (math.ceil(size[0]/3), math.ceil(size[1]/3)), 
            (math.ceil(size[0]/4), math.ceil(size[1]/4)),
            # square
            (math.ceil(size[0]/2), math.ceil(size[0]/2)), 
            (math.ceil(size[0]/3), math.ceil(size[0]/3)), 
            (math.ceil(size[0]/4), math.ceil(size[0]/4)),
            (math.ceil(size[0]/5), math.ceil(size[0]/5)),
            (math.ceil(size[0]/8), math.ceil(size[0]/8)),
            (math.ceil(size[0]/16), math.ceil(size[0]/16)),
            (math.ceil(size[0]/16), math.ceil(size[0]/32))
        ])
    geometry = ImageProcessorPipeline(size=size)
    geometry.addTask(DrawCheckerboardProcessor(size = blocksize, start = start, fill=foreground))
    geometry.addTask(FillBackgroundProcessor(background = background))
    return geometry