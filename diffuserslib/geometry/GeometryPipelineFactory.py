from .GeometryPipeline import GeometryPipeline
from .GeometryTasks import DrawRegularShape, DrawCheckerboard, Symmetrize, Spiralize, RandomPositionArgument
from .. import RandomNumberArgument, RandomChoiceArgument

import math


def shapeGeometryPipeline(size=(512, 512), sides=RandomNumberArgument(3, 6), minsize=32, maxsize=256, shapes=1,
                          symmetry=RandomChoiceArgument(["horizontal", "vertical", "rotation", "none"])):
    geometry = GeometryPipeline(size=size)
    for i in range(shapes):
        geometry.addTask(DrawRegularShape(
            position=RandomPositionArgument(), 
            size=RandomNumberArgument(minsize, maxsize),
            sides=sides
        ))
    geometry.addTask(Symmetrize(
        type=symmetry
    ))
    return geometry


def spiralGeometryPipeline(size=(512, 512), sides=RandomNumberArgument(3, 6), minsize=32, maxsize=256, shapes=1, rotation=360, steps=8, zoom=4):
    geometry = GeometryPipeline(size=size)
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


def checkerboardGeometryPipeline(size=(512, 512), blocksize = None):
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
            (math.ceil(size[0]/8), math.ceil(size[0]/8))
        ])
    geometry = GeometryPipeline(size=size)
    geometry.addTask(DrawCheckerboard(size = blocksize))
    return geometry