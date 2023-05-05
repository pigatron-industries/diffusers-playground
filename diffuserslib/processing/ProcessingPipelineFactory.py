from .ProcessingPipeline import ImageProcessorPipeline
from .ImageProcessor import InitImageProcessor, FillBackgroundProcessor, ResizeProcessor, CropProcessor
from .TransformProcessors import SimpleTransformProcessor, SymmetrizeProcessor, SpiralizeProcessor
from .GeometryProcessors import DrawRegularShapeProcessor, DrawCheckerboardProcessor, DrawGeometricSpiralProcessor
from ..batch import RandomNumberArgument, RandomChoiceArgument, RandomPositionArgument, RandomNumberTuple

import math


class ProcessingPipelineBuilder(ImageProcessorPipeline):
    def __init__(self, size=None, oversize=0):
        super().__init__(size, oversize)


    @classmethod
    def fromImage(cls, image):
        pipeline = cls(oversize=0)
        pipeline.addTask(InitImageProcessor(image=image))
        return pipeline
    

    @classmethod
    def fromBlank(cls, size=RandomChoiceArgument([(512, 768), (768, 512), (512, 512)])):
        pipeline = cls(size=size)
        return pipeline
        

    def fillBackground(self, background = "black"):
        self.addTask(FillBackgroundProcessor(background = background))
        return self
    

    def drawGeometricSpiral(self, outline="white", fill="black", iterations = 14, 
                        direction = RandomChoiceArgument(["up", "down", "left", "right"]),
                        turn=RandomChoiceArgument(["clockwise", "anticlockwise"]),
                        draw=RandomChoiceArgument([(False, True), (True, True), (True, False)]),
                        ratio = 1/1.618033988749895, rect = (0, 0, 1, 1)):
        self.addTask(DrawGeometricSpiralProcessor(iterations = iterations, 
                                                    direction = direction, turn=turn,
                                                    draw = draw, outline = outline, fill = fill, ratio = ratio, 
                                                    rect = rect))
        return self


    def simpleTransform(self, transform=RandomChoiceArgument(["fliphorizontal", "flipvertical", "rotate90", "rotate180", "rotate270", "none"])):
        self.addTask(SimpleTransformProcessor(type=transform))
        return self


    def resize(self, resizetype=RandomChoiceArgument(["stretch", "extend"]), size=RandomChoiceArgument([(512, 768), (768, 512)]),
                        halign=RandomChoiceArgument(["left", "right", "centre"]), valign=RandomChoiceArgument(["top", "bottom", "centre"]), fill="black"):
        self.addTask(ResizeProcessor(type=resizetype, size=size, fill=fill, halign=halign, valign=valign))
        return self


    def crop(self, size = RandomChoiceArgument([(512, 768), (768, 512)]), position = RandomNumberTuple(2, 0.0, 1.0)):
        self.addTask(CropProcessor(size = size, position = position))
        return self


    def drawShapeGeometry(self, foreground="black", outline=None, sides=RandomNumberArgument(3, 6), minsize=32, maxsize=256, shapes=1):
        for i in range(shapes):
            self.addTask(DrawRegularShapeProcessor(
                position=RandomPositionArgument(), 
                size=RandomNumberArgument(minsize, maxsize),
                sides=sides,
                fill=foreground,
                outline=outline
            ))
        return self
    

    def symmetrize(self, symmetry=RandomChoiceArgument(["horizontal", "vertical", "rotation", "none"])):
        self.addTask(SymmetrizeProcessor(type=symmetry))
        return self


    def spiralize(self, rotation=360, steps=8, zoom=4):
        self.addTask(SpiralizeProcessor(rotation = rotation, steps = steps, zoom = zoom))
        return self
    

    def drawCheckerboard(self, size=(512, 512), background="white", foreground="black", blocksize = None, start = "black"):
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
        self.addTask(DrawCheckerboardProcessor(size = blocksize, start = start, fill=foreground))
        self.addTask(FillBackgroundProcessor(background = background))
        return self

