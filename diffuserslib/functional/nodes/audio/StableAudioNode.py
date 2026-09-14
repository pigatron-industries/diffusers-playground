import random

import numpy as np
import torch

from diffuserslib.functional.types import *
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.GlobalConfig import GlobalConfig

MAX_SEED = 4294967295

# Production Stable Audio 3 checkpoint (distilled, 8-step ping-pong sampling,
# classifier-free guidance baked into the weights). For prompt-following with
# classifier-free guidance use "stabilityai/stable-audio-3-medium-base".
DEFAULT_MODEL = "/Volumes/T9/models/other/stable-audio-medium"


class StableAudioNode(FunctionalNode):
    """Text-to-audio (and audio-to-audio) generation using Stable Audio 3.

    Uses diffusers' ``StableAudio3Pipeline`` for plain text-to-audio and
    ``StableAudio3AudioToAudioPipeline`` when a reference ``initaudio`` clip is
    supplied (conditioned variation via ``noise_level``).

    Parameter notes:
    - ``steps``: ``0`` (or ``None``) lets the pipeline pick its native step
      count (8 for the distilled model, ~100 for the base model). Pass a
      positive integer to override.
    - ``cfg_scale`` / ``negative_prompt``: only meaningful for the non-distilled
      ``stable-audio-3-medium-base`` checkpoint. The distilled default ignores
      both (guidance is baked into the weights).
    - ``model``: any diffusers-format Stable Audio 3 checkpoint (repo id or a
      local directory produced by diffusers' conversion script).
    """

    def __init__(self,
                 prompt:StringFuncType = "",
                 negative_prompt:StringFuncType = "",
                 duration:FloatFuncType = 10.0,
                 steps:IntFuncType = 0,
                 cfg_scale:FloatFuncType = 1.0,
                 seed:IntFuncType = 0,
                 initaudio:AudioFuncType|None = None,
                 noise_level:FloatFuncType = 1.0,
                 model:StringFuncType = DEFAULT_MODEL,
                 name:str="stableaudio"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.addParam("negative_prompt", negative_prompt, str)
        self.addParam("duration", duration, float)
        self.addParam("steps", steps, int)
        self.addParam("cfg_scale", cfg_scale, float)
        self.addParam("seed", seed, int)
        self.addParam("initaudio", initaudio, Audio)
        self.addParam("noise_level", noise_level, float)
        self.addParam("model", model, str)
        # Cache pipelines keyed by (model id, audio-to-audio flag) so repeated
        # runs reuse the loaded checkpoint and swapping models reloads cleanly.
        self._pipeline = None
        self._loaded_key = None


    def process(self, prompt:str, negative_prompt:str, duration:float, steps:int, cfg_scale:float,
                seed:int|None, initaudio:Audio|None, noise_level:float, model:str):
        if seed is None:
            seed = random.randint(0, MAX_SEED)

        audio2audio = initaudio is not None
        pipeline = self._load_pipeline(model, audio2audio)

        generator = torch.Generator("cpu").manual_seed(seed)
        # 0 / None -> let the pipeline use its native step count.
        num_inference_steps = steps if steps and steps > 0 else None

        if audio2audio:
            reference = self._prepare_reference_audio(pipeline, initaudio)
            output = pipeline(
                prompt=prompt,
                duration=duration,
                audio=reference,
                init_noise_level=noise_level,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).audios
        else:
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                duration=duration,
                guidance_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).audios

        return self._to_audio(output, pipeline.vae.config.sampling_rate)


    def _load_pipeline(self, model:str, audio2audio:bool):
        key = (model, audio2audio)
        if self._pipeline is not None and self._loaded_key == key:
            return self._pipeline

        try:
            from diffusers import StableAudio3Pipeline, StableAudio3AudioToAudioPipeline
        except ImportError as exc:
            raise ImportError(
                "Stable Audio 3 requires diffusers >= 0.40. "
                "Upgrade with `pip install -U diffusers`."
            ) from exc

        device = GlobalConfig.device
        # float16 on MPS produces noise (per the SA3 docs); keep float32 on
        # CPU/MPS and only drop to float16 on CUDA where it is safe.
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

        pipeline_cls = StableAudio3AudioToAudioPipeline if audio2audio else StableAudio3Pipeline
        pipeline = pipeline_cls.from_pretrained(model, torch_dtype=dtype)
        pipeline = pipeline.to(device)

        self._pipeline = pipeline
        self._loaded_key = key
        return pipeline


    def _prepare_reference_audio(self, pipeline, initaudio:Audio):
        import torchaudio

        target_sr = int(pipeline.vae.config.sampling_rate)
        wave = torch.from_numpy(np.asarray(initaudio.audio_array)).float()
        if wave.ndim == 1:
            wave = wave.unsqueeze(0)  # (samples) -> (1, samples)
        wave = wave.unsqueeze(0)  # (channels, samples) -> (1, channels, samples)

        source_sr = int(initaudio.sample_rate)
        if source_sr != target_sr:
            wave = torchaudio.functional.resample(wave, orig_freq=source_sr, new_freq=target_sr)
        return wave.to(pipeline.device)


    def _to_audio(self, audios:torch.Tensor, sample_rate:int) -> Audio:
        # audios: (batch, channels, samples) -> take the first waveform
        waveform = audios[0]
        waveform = waveform.clamp(-1.0, 1.0).to(torch.float32).cpu()
        if waveform.shape[0] == 1:
            audio_array = waveform[0].numpy()  # (samples,)
        else:
            audio_array = waveform.T.numpy()  # (samples, channels)
        return Audio(audio_array, sample_rate)
