from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.types import Vector, VectorsFuncType
from PIL import ImageDraw, Image
from typing import List, Tuple, Callable, Dict, Any
from scipy.spatial import Voronoi
import math
import numpy as np
import itertools

DrawOptionsType = Tuple[bool, bool, bool]
DrawOptionsFuncType = DrawOptionsType | Callable[[], DrawOptionsType]


class DrawVoronoiNode(FunctionalNode):
    def __init__(self, 
                 image: ImageFuncType,
                 points: VectorsFuncType,
                 outline_colour: ColourFuncType = "white", 
                 point_colour: ColourFuncType = "white",
                 draw_options: DrawOptionsFuncType = (True, True, True),  # (bounded lines, unbounded lines, points)
                 line_probability: FloatFuncType = 1.0,
                 radius: IntFuncType = 2,
                 name:str = "draw_voronoi"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("points", points, List[Vector])
        self.addParam("outline_colour", outline_colour, ColourType)
        self.addParam("point_colour", point_colour, ColourType)
        self.addParam("draw_options", draw_options, DrawOptionsType)
        self.addParam("radius", radius, int)
        self.addParam("line_probability", line_probability, float)


    def process(self, image: Image.Image,
                points: List[Vector],
                outline_colour: ColourType, 
                point_colour: ColourType,
                draw_options: DrawOptionsType,
                line_probability: float = 1,
                radius: float = 2) -> Image.Image:
        drawBoundedLines, drawUnboundedLines, drawPoints = draw_options
        points_array = np.array([vector.coordinates for vector in points]) * image.size
        image = image.copy()

        lines = self.getLines(points_array, boundedLines=drawBoundedLines, unboundedLines=drawUnboundedLines)
        draw = ImageDraw.Draw(image)

        for line in lines:
            if (np.random.random() < line_probability):
                draw.line(line, fill=outline_colour, width=1)

        if (drawPoints):
            for point in points_array:
                draw.ellipse((point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius), fill=point_colour)
            
        return image
    

    def getLines(self, points:np.ndarray, boundedLines:bool = True, unboundedLines:bool = True) -> List[Tuple[float, float, float, float]]:
        voronoi = Voronoi(points)
        centre = voronoi.points.mean(axis=0)
        lines = []

        for ridge_points, ridge_vertices in voronoi.ridge_dict.items():
            if boundedLines and -1 not in ridge_vertices:
                # finite ridge between two vertices
                vert1 = voronoi.vertices[ridge_vertices[0]]
                vert2 = voronoi.vertices[ridge_vertices[1]]
                line = (vert1[0], vert1[1], vert2[0], vert2[1])
                lines.append(line)
            elif unboundedLines:
                # infinite ridge
                vert = voronoi.vertices[ridge_vertices[1]]
                point1 = voronoi.points[ridge_points[0]]
                point2 = voronoi.points[ridge_points[1]]
                midpoint = (point2 + point1) * 0.5
                normal = (midpoint - vert) / np.linalg.norm(midpoint - vert)
                direction = midpoint - centre
                normal = np.array([math.copysign(normal[0], direction[0]), math.copysign(normal[1], direction[1])])

                endpoint = vert + (normal * 512) # make line go twice as far as midpoint
                line = (vert[0], vert[1], endpoint[0], endpoint[1])
                lines.append(line)

        return lines