from typing import Callable, Tuple, List
from PIL import Image

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