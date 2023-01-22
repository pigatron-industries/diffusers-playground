from .GeometryPipeline import GeometryPipeline
from .GeometryTasks import DrawRegularShape, Symmetrize, Spiralize, RandomPositionArgument
from .. import RandomNumberArgument, RandomStringArgument


def shapeGeometryPipeline(size=(512, 512), sides=RandomNumberArgument(3, 6), minsize=32, maxsize=256, shapes=1,
                          symmetry=RandomStringArgument(["horizontal", "vertical", "rotation", "none"])):
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