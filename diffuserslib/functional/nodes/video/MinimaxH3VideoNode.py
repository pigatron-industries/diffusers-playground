import os
import tempfile

from PIL import Image

from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.types import Video
from diffuserslib.util.CommandProcess import CommandProcess
from diffuserslib.GlobalConfig import GlobalConfig


class MinimaxH3VideoNode(FunctionalNode):
    """Text-to-video (and image-conditioned video) generation using MiniMax-H3 on Apple Silicon.

    Shells out to the ``minimax-h3-mac`` generation CLI
    (``{GlobalConfig.minimax_h3_dir}/scripts/generate.py``) using the python
    interpreter at ``GlobalConfig.minimax_h3_python``.

    Model locations are configurable:
    - The base checkout (``GlobalConfig.minimax_h3_dir``) and interpreter
      (``GlobalConfig.minimax_h3_python``) are read from ``GlobalConfig`` so a
      single knob moves every model path at once.
    - Each model path (``checkpoint``, ``transformer``, ``text_encoder``,
      ``turbo_lora``) is an optional node parameter; leave it as ``None`` to
      fall back to the default location under ``{workdir}/models/...``.

    Keyframe conditioning: the H3 CLI takes keyframes via a repeatable
    ``--image`` flag paired with ``--anchor {first|last}`` (in order). ``first_image``
    and ``last_image`` are written to temp files and mapped onto that contract.
    """

    # Default model locations relative to the minimax-h3-mac checkout.
    DEFAULT_CHECKPOINT = os.path.join("MiniMax-H3", "FL2VA")
    DEFAULT_TRANSFORMER = os.path.join("MiniMax-H3-MLX-Argus-Calibrated-INT8")
    DEFAULT_TEXT_ENCODER = os.path.join("MiniMax-H3-MLX-TextEncoder-4bit")
    DEFAULT_TURBO_LORA = os.path.join("MiniMax-H3-Turbo-v4-step600-EMA-MLX")

    def __init__(self,
                 prompt:StringFuncType = "",
                 first_image:ImageFuncType|None = None,
                 last_image:ImageFuncType|None = None,
                 resolution:SizeFuncType = (768, 448),
                 duration:FloatFuncType = 5.0,
                 steps:IntFuncType = 9,
                 seed:IntFuncType|None = None,
                 name:str = "minimax_h3_video"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.addParam("first_image", first_image, Image.Image)
        self.addParam("last_image", last_image, Image.Image)
        self.addParam("resolution", resolution, SizeType)
        self.addParam("duration", duration, float)
        self.addParam("steps", steps, int)
        self.addParam("seed", seed, int)


    def process(self,
                prompt:str,
                first_image:Image.Image|None,
                last_image:Image.Image|None,
                resolution:SizeType,
                duration:float,
                steps:int,
                seed:int|None) -> Video:
        workdir = GlobalConfig.minimax_h3_proj_dir
        modeldir = GlobalConfig.minimax_h3_model_dir
        python = os.path.join(workdir, ".venv", "bin", "python")

        checkpoint = os.path.join(modeldir, self.DEFAULT_CHECKPOINT)
        transformer = os.path.join(modeldir, self.DEFAULT_TRANSFORMER)
        text_encoder = os.path.join(modeldir, self.DEFAULT_TEXT_ENCODER)
        turbo_lora = os.path.join(modeldir, self.DEFAULT_TURBO_LORA)

        out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=True)

        first_path = last_path = None
        if first_image is not None:
            first_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            first_image.save(first_path)
        if last_image is not None:
            last_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            last_image.save(last_path)

        command = [
            python, f"{workdir}/scripts/generate.py", f'"{prompt}"',
            "--checkpoint", checkpoint,
            "--transformer", transformer,
            # "--text-encoder", text_encoder,
            "--turbo-lora", turbo_lora,
            "--turbo-lora-scale", str(1.0),
            "--resolution", f"{resolution[0]}x{resolution[1]}",
            "--duration", str(duration),
            "--steps", str(steps),
            "--low-memory", 
            "--stream-blocks",
            "--output", out_file.name,
        ]
        if seed is not None:
            command += ["--seed", str(seed)]
        if first_path is not None:
            command += ["--image", first_path, "--anchor", "first"]
        if last_path is not None:
            command += ["--image", last_path, "--anchor", "last"]

        process = CommandProcess(command)
        process.runSync()   # raises on non-zero exit, per CommandProcess.run()

        return Video(file=out_file)
