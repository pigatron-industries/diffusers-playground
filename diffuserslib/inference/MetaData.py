import piexif
import json
import random
from datetime import datetime
from PIL import Image

MAX_SEED = 4294967295


def build_civitai_metadata(params, seed: int | None = None) -> str:
    """Build a Civitai-style metadata string from GenerationParameters-like object."""
    prompt = getattr(params, 'original_prompt', None) or getattr(params, 'prompt', None) or ""
    negative_prompt = getattr(params, 'original_negprompt', None) or getattr(params, 'negprompt', None) or ""
    steps = getattr(params, 'steps', None)
    sampler = getattr(params, 'scheduler', None)
    cfg_scale = getattr(params, 'cfgscale', None)
    if seed is None:
        seed = params.seed if getattr(params, 'seed', None) is not None else random.randint(0, MAX_SEED)
    size = f"{getattr(params,'width',0)}x{getattr(params,'height',0)}"
    clip_skip = getattr(params, 'clipskip', None)
    created_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    resources = []
    try:
        if hasattr(params, 'modelConfig') and params.modelConfig:
            model = params.modelConfig[0]
            model_name = None
            model_version = None
            if getattr(model, 'data', None) and isinstance(model.data, dict):
                model_name = model.data.get('name') or model.modelid
                model_version = model.data.get('version') or model.revision
            else:
                model_name = model.modelid
                model_version = model.revision
            resources.append({
                "type": "checkpoint",
                "modelVersionId": model.modelid,
                "modelName": model_name,
                "modelVersionName": model_version
            })
    except Exception:
        pass

    # Add LORA resources from params.loras
    try:
        for lora in getattr(params, 'loras', []) or []:
            resources.append({
                "type": "lora",
                "weight": getattr(lora, 'weight', 1.0),
                "modelVersionId": getattr(lora, 'name', None) or getattr(lora, 'modelid', None) or "",
                "modelName": getattr(lora, 'name', None) or "",
                "modelVersionName": ""
            })
    except Exception:
        pass

    metadata_string = f"{prompt}Negative prompt: {negative_prompt}Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg_scale}, Seed: {seed}, Size: {size}, Clip skip: {clip_skip}, Created Date: {created_date}, Civitai resources: {json.dumps(resources, separators=(',', ':'))}"
    return metadata_string


def add_ai_metadata_to_pil_image(image: Image.Image, metadata_string: str) -> Image.Image:
    """Embed AI generation metadata into a PIL Image's EXIF UserComment and ImageDescription.

    This sets `image.info['exif']` with the EXIF bytes so that saving the image with Pillow
    and passing `exif=image.info['exif']` will include the metadata.
    """
    try:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        if hasattr(image, 'info') and 'exif' in image.info and image.info['exif']:
            try:
                exif_dict = piexif.load(image.info['exif'])
            except Exception:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        user_comment = metadata_string.encode('utf-8')
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = metadata_string.encode('utf-8')
        exif_dict["0th"][piexif.ImageIFD.Software] = "9ab6306e-d9fb-4777-84ee-6434e564edde"
        exif_dict["0th"][piexif.ImageIFD.Artist] = "ai"
        exif_bytes = piexif.dump(exif_dict)
        if not hasattr(image, 'info'):
            image.info = {}
        image.info['exif'] = exif_bytes

    except Exception:
        pass
    return image
