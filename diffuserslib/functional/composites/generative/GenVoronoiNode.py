from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from diffuserslib.functional.elements import *
from PIL import Image


class GenVoronoiNode(FunctionalNode):
    def __init__(self, 
                 size: SizeFuncType,
                 points: Points2DFuncType,
                 draw_options: DrawOptionsFuncType = (True, True, True),  # (bounded lines, unbounded lines, points)
                 line_probablity: FloatFuncType = 1,
                 radius: IntFuncType = 2,
                 name:str = "gen_voronoi"):
        super().__init__(name)
        self.addParam("size", size, TypeInfo(ParamType.IMAGE_SIZE))
        self.addParam("points", points, TypeInfo(ParamType.POINT2D, multiple=True))
        self.addParam("draw_options", draw_options, TypeInfo(ParamType.BOOL, size=3))
        self.addParam("radius", radius, TypeInfo(ParamType.INT))
        self.addParam("line_probablity", line_probablity, TypeInfo(ParamType.FLOAT, restrict_num=(0.0, 1.0, 0.1)))

        self.image = NewImageNode(size=size, background_colour=(0, 0, 0))
        self.voronoi = DrawVoronoiNode(image = self.image, points = points)


    def process(self, size: SizeType,
                points: Points2DType,
                draw_options: DrawOptionsType,
                line_probablity: float,
                radius: float) -> Image.Image:
        self.image.setParam("size", size)
        self.voronoi.setParam("points", points)
        self.voronoi.setParam("draw_options", draw_options)
        self.voronoi.setParam("line_probablity", line_probablity)
        self.voronoi.setParam("radius", radius)
        return self.voronoi()
    