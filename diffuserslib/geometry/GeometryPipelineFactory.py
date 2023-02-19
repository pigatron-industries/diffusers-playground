from .GeometryPipeline import GeometryPipeline
from .GeometryTasks import *
from .. import RandomNumberArgument, RandomChoiceArgument

import math


def simpleTransform(image, transform=RandomChoiceArgument(["fliphorizontal", "flipvertical", "rotate90", "rotate180", "rotate270", "none"])):
    geometry = GeometryPipeline(size=image.size)
    geometry.addTask(DrawImageTask(image))
    geometry.addTask(SimpleTransformTask(type=transform))
    return geometry


def shapeGeometryPipeline(size=(512, 512), background="white", foreground="black", sides=RandomNumberArgument(3, 6), minsize=32, maxsize=256, shapes=1,
                          symmetry=RandomChoiceArgument(["horizontal", "vertical", "rotation", "none"])):
    geometry = GeometryPipeline(size=size, background=background)
    for i in range(shapes):
        geometry.addTask(DrawRegularShape(
            position=RandomPositionArgument(), 
            size=RandomNumberArgument(minsize, maxsize),
            sides=sides,
            fill=foreground
        ))
    geometry.addTask(Symmetrize(
        type=symmetry
    ))
    return geometry


def spiralGeometryPipeline(size=(512, 512), background="white", sides=RandomNumberArgument(3, 6), minsize=32, maxsize=256, shapes=1, rotation=360, steps=8, zoom=4):
    geometry = GeometryPipeline(size=size, background=background)
    for i in range(shapes):
        geometry.addTask(DrawRegularShape(
            position=RandomPositionArgument(), 
            size=RandomNumberArgument(minsize, maxsize),
            sides=sides
        ))
    geometry.addTask(Spiralize(
        rotation = rotation, steps = steps, zoom = zoom
    ))
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
    geometry = GeometryPipeline(size=size, background=background)
    geometry.addTask(DrawCheckerboard(size = blocksize, start = start, fill=foreground))
    return geometry