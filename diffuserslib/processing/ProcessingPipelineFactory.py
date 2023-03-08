from .ImageProcessor import ImageProcessorPipeline, InitImageProcessor, FillBackgroundProcessor
from .TransformProcessors import SimpleTransformProcessor, SymmetrizeProcessor, SpiralizeProcessor
from .GeometryProcessors import DrawRegularShapeProcessor, DrawCheckerboardProcessor
from ..batch import RandomNumberArgument, RandomChoiceArgument, RandomPositionArgument

import math


def initImage(image):
    pipeline = ImageProcessorPipeline(oversize=0)
    pipeline.addTask(InitImageProcessor(image))
    return pipeline


def simpleTransform(image, transform=RandomChoiceArgument(["fliphorizontal", "flipvertical", "rotate90", "rotate180", "rotate270", "none"])):
    pipeline = ImageProcessorPipeline()
    pipeline.addTask(InitImageProcessor(image))
    pipeline.addTask(SimpleTransformProcessor(type=transform))
    return pipeline


def shapeGeometryPipeline(size=(512, 512), background="white", foreground="black", outline=None, sides=RandomNumberArgument(3, 6), minsize=32, maxsize=256, shapes=1,
                          symmetry=RandomChoiceArgument(["horizontal", "vertical", "rotation", "none"])):
    pipeline = ImageProcessorPipeline(size=size)
    for i in range(shapes):
        pipeline.addTask(DrawRegularShapeProcessor(
            position=RandomPositionArgument(), 
            size=RandomNumberArgument(minsize, maxsize),
            sides=sides,
            fill=foreground,
            outline=outline
        ))
    pipeline.addTask(SymmetrizeProcessor(type=symmetry))
    pipeline.addTask(FillBackgroundProcessor(background = background))
    return pipeline


def spiralGeometryPipeline(size=(512, 512), background="white", fill="black", outline=None, sides=RandomNumberArgument(3, 6), minsize=32, maxsize=256, shapes=1, rotation=360, steps=8, zoom=4):
    pipeline = ImageProcessorPipeline(size=size)
    for i in range(shapes):
        pipeline.addTask(DrawRegularShapeProcessor(
            position=RandomPositionArgument(), 
            size=RandomNumberArgument(minsize, maxsize),
            sides=sides,
            fill=fill,
            outline=outline
        ))
    pipeline.addTask(SpiralizeProcessor(
        rotation = rotation, steps = steps, zoom = zoom
    ))
    pipeline.addTask(FillBackgroundProcessor(background = background))
    return pipeline


def checkerboardGeometryPipeline(size=(512, 512), background="white", foreground="black", blocksize = None, start = "black"):
    if(blocksize is None):
        blocksize = RandomChoiceArgument([
            #  aspect ratio dependent
            (math.ceil(size[0]/1), math.ceil(size[1]/2)), 
            (math.ceil(size[0]/2), math.ceil(size[1]/1)), 
            (math.ceil(size[0]/2), math.ceil(size[1]/2)), 
            (math.ceil(size[0]/1), math.ceil(size[1]/3)), 
            (math.ceil(size[0]/3), math.ceil(size[1]/1)), 
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
    pipeline = ImageProcessorPipeline(size=size)
    pipeline.addTask(DrawCheckerboardProcessor(size = blocksize, start = start, fill=foreground))
    pipeline.addTask(FillBackgroundProcessor(background = background))
    return pipeline