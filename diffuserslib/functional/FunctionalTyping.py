from typing import Callable, Tuple, List, Any
from dataclasses import dataclass
from PIL import Image

# Python type definitions for node inputs
StringFuncType = str | Callable[[], str]
IntFuncType = int | Callable[[], int]
FloatFuncType = float | Callable[[], float]

StringsFuncType = List[str] | Callable[[], List[str]]
IntsFuncType = List[int] | Callable[[], List[int]]
FloatsFuncType = List[float] | Callable[[], List[float]]

SizeType = Tuple[int, int]
SizeFuncType = SizeType | Callable[[], SizeType]

Point2DType = Tuple[float, float]
Point2DFuncType = Point2DType | Callable[[], Point2DType]
Points2DType = List[Point2DType]
Points2DFuncType = Points2DType | Callable[[], Points2DType]

ImageFuncType = Image.Image | Callable[[], Image.Image]

ColourType = Tuple[int, int, int] | str
ColourFuncType = ColourType | Callable[[], ColourType]


# workflow paramter type information
@dataclass
class TypeInfo:
    type: str|None = None
    restrict_num: Tuple[float, float, float]|None = None
    restrict_choice: List[Any]|None = None
    size: int|None = None
    multiple: bool = False


class ParamType:
    STRING = "String"
    INT = "Int"
    FLOAT = "Float"
    BOOL = "Bool"
    COLOUR = "Colour"
    POINT2D = "Point2D"
    IMAGE_SIZE = "ImageSize"
    IMAGE = "Image"
