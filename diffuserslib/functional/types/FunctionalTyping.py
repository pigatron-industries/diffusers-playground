from typing import Callable, Tuple, List, Any
from PIL import Image

# Python type definitions for node inputs
StringFuncType = str | Callable[[], str]
IntFuncType = int | Callable[[], int]
FloatFuncType = float | Callable[[], float]
BoolFuncType = bool | Callable[[], bool]

StringsFuncType = List[str] | Callable[[], List[str]]
IntsFuncType = List[int] | Callable[[], List[int]]
FloatsFuncType = List[float] | Callable[[], List[float]]
BoolsFuncType = List[bool] | Callable[[], List[bool]]

SizeType = Tuple[int, int]
SizeFuncType = SizeType | Callable[[], SizeType]
FloatTupleType = Tuple[float, float]
FloatTupleFuncType = FloatTupleType | Callable[[], FloatTupleType]

MinMaxIntType = Tuple[int, int]
MinMaxIntFuncType = MinMaxIntType | Callable[[], MinMaxIntType]
MinMaxFloatType = Tuple[float, float]
MinMaxFloatFuncType = MinMaxFloatType | Callable[[], MinMaxFloatType]

Point2DType = Tuple[float, float]
Point2DFuncType = Point2DType | Callable[[], Point2DType]
Points2DType = List[Point2DType]
Points2DFuncType = Points2DType | Callable[[], Points2DType]

RectType = Tuple[float, float, float, float]
RectFuncType = RectType | Callable[[], RectType]

ImageFuncType = Image.Image | Callable[[], Image.Image]
FramesFuncType = List[Image.Image] | Callable[[], List[Image.Image]]

ColourType = Tuple[int, int, int] | str
ColourFuncType = ColourType | Callable[[], ColourType]

ListType = List[Any]
ListFuncType = ListType | Callable[[], ListType]