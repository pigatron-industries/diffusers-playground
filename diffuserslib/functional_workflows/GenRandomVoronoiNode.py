from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from diffuserslib.functional import *
from PIL import Image

name = "Random Voronoi"

def build():
    return GenRandomVoronoiNode()


class GenRandomVoronoiNode(FunctionalNode):
    def __init__(self, 
                 size: SizeFuncType = (512, 512),
                 num_points:int = 20,
                 draw_options: DrawOptionsFuncType = (True, True, True),  # (bounded lines, unbounded lines, points)
                 line_probablity: FloatFuncType = 1,
                 radius: IntFuncType = 2,
                 name:str = "gen_voronoi"):
        super().__init__(name)
        self.addParam("size", size, TypeInfo(ParamType.IMAGE_SIZE))
        self.addParam("num_points", num_points, TypeInfo(ParamType.INT))
        self.addParam("draw_options", draw_options, TypeInfo(ParamType.BOOL, size=3))
        self.addParam("radius", radius, TypeInfo(ParamType.INT))
        self.addParam("line_probablity", line_probablity, TypeInfo(ParamType.FLOAT, restrict_num=(0.0, 1.0, 0.1)))

        self.random_points = RandomPoints2DNode(num_points=num_points)
        self.voronoi = GenVoronoiNode(size=size, points=self.random_points)


    def process(self, size: SizeType,
                num_points: int,
                draw_options: DrawOptionsType,
                line_probablity: float,
                radius: float) -> Image.Image:
        print(line_probablity)
        self.random_points.setParam("num_points", num_points)
        self.voronoi.setParam("size", size)
        self.voronoi.setParam("draw_options", draw_options)
        self.voronoi.setParam("line_probablity", line_probablity)
        self.voronoi.setParam("radius", radius)
        return self.voronoi()
    