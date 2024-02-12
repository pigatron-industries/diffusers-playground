from diffuserslib.functional import *

def name():
    return "Random Voronoi"

def build():
    size_user_input = SizeUserInputNode(value = (512, 512), name = "size")
    num_points_input = IntUserInputNode(value = 20, name = "num_points")

    random_points = RandomPoints2DNode(num_points = num_points_input)
    new_image = NewImageNode(size = size_user_input, background_colour = (0, 0, 0))
    voronoi = DrawVoronoiNode(image = new_image, points = random_points)

    return voronoi




# class GenRandomVoronoiNode(FunctionalNode):
#     name = "Random Voronoi"

#     def __init__(self, 
#                  size: SizeFuncType = (512, 512),
#                  num_points:int = 20,
#                  draw_options: DrawOptionsFuncType = (True, True, True),  # (bounded lines, unbounded lines, points)
#                  line_probablity: FloatFuncType = 1,
#                  radius: IntFuncType = 2,
#                  name:str = "gen_voronoi"):
#         super().__init__(name)
#         self.addParam("size", size, TypeInfo(ParamType.IMAGE_SIZE))
#         self.addParam("num_points", num_points, TypeInfo(ParamType.INT))
#         self.addParam("draw_options", draw_options, TypeInfo(ParamType.BOOL, size=3))
#         self.addParam("radius", radius, TypeInfo(ParamType.INT))
#         self.addParam("line_probablity", line_probablity, TypeInfo(ParamType.FLOAT, restrict_num=(0.0, 1.0, 0.1)))

#         self.random_points = RandomPoints2DNode(num_points=num_points)
#         self.voronoi = GenVoronoiNode(size=size, points=self.random_points)


#     def process(self, size: SizeType,
#                 num_points: int,
#                 draw_options: DrawOptionsType,
#                 line_probablity: float,
#                 radius: float) -> Image.Image:
#         self.random_points.setParam("num_points", num_points)
#         self.voronoi.setParam("size", size)
#         self.voronoi.setParam("draw_options", draw_options)
#         self.voronoi.setParam("line_probablity", line_probablity)
#         self.voronoi.setParam("radius", radius)
#         return self.voronoi()
    