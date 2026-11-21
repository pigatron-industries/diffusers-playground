from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters
from diffusers.models.modeling_outputs import Transformer2DModelOutput
import sys
import types
import torch


def _makeMpsSafeForward(validate_skip_layers):
    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=None, return_dict=True, skip_layers=None):
        # MPS-safe copy of MOSSImageTransformer.forward from the model repo's pipeline_moss_image.py.
        # The original computes the timestep sinusoid in float64 on-device, which MPS does not support;
        # here freq is built on CPU (float64) and moved to device as float32, with phase/sin/cos in fp32
        # (matching the repo's stated first-release defaults). Keep in sync with MOSSImageTransformer.forward.
        skip_layers = validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
            raise ValueError('hidden_states must be [B, latent_channels, H, W].')
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError('encoder_hidden_states must be [B, S, text_dim].')
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('encoder_attention_mask must be a binary [B, S] mask.')
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        # Training computes the sinusoid in float64 and feeds float32 into its MLP.
        half = self.config.freq_dim // 2
        freq = (10000 ** (-torch.arange(half, dtype=torch.float64) / half)).to(times.device, torch.float32)
        phase = times[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index:index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError('Each sample needs at least one valid text token.')
            video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
            # Match the reference activation dtype under fp32 weights + autocast.
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width:])
            x = torch.cat((video, text), 1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    x = layer(x, rope, mod[index:index + 1])
            x = self.final_norm_video(x[:, :height * width].to(self.final_norm_video.weight.dtype))
            x = x * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        sample = torch.cat(outputs)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)
    return forward


def patchMossTransformerForMps(transformer):
    """Bind an MPS-safe forward onto the MOSSImageTransformer instance (idempotent)."""
    if getattr(transformer, "_moss_mps_safe_forward", False):
        return
    module = sys.modules[type(transformer).__module__]
    transformer.forward = types.MethodType(_makeMpsSafeForward(module._validate_skip_layers), transformer)
    transformer._moss_mps_safe_forward = True


class MossImagePipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, params:GenerationParameters, device, **kwargs):
        self.safety_checker = params.safetychecker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        super().__init__(params, inferencedevice, cls, dtype=None, **kwargs)

    def createPipeline(self, params:GenerationParameters, cls):
        pipeline_params = self.createPipelineParams(params)
        from diffusers import DiffusionPipeline
        if(type(cls) is str):
            pipeline_params['custom_pipeline'] = cls
            cls = DiffusionPipeline

        for i, modelConfig in enumerate(params.modelConfig):
            pipeline = self.loadPipeline(modelConfig, cls, pipeline_params)
            if(i == 0):
                self.pipeline = pipeline
            else:
                self.mergeModel(pipeline, params.models[i].weight)
        patchMossTransformerForMps(self.pipeline.transformer)

    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        # MOSS-Image ships its own pipeline code (pipeline_moss_image.py) in the model repo;
        # transformer/text encoder run bf16 while the FLUX.2 VAE decodes in fp32
        pipeline_params['trust_remote_code'] = True
        pipeline_params['torch_dtype'] = {"default": torch.bfloat16, "vae": torch.float32}
        return pipeline_params

    def diffusers_inference(self, prompt, negative_prompt, seed, guidance_scale=3.0, scheduler=None, clip_skip=None, **kwargs):
        # MOSS-Image is a flow-matching model; its custom scheduler (shift 5) comes from the repo
        generator, seed = self.createGenerator(seed)
        output = self.pipeline(prompt=prompt, negative_prompt=negative_prompt, generator=generator, guidance_scale=guidance_scale, return_dict=True, **kwargs)
        return output, seed


class MossImageGeneratePipelineWrapper(MossImagePipelineWrapper):

    PIPELINE_MAP = {
        #img2img,  inpaint
        (False,     False):    "pipeline_moss_image",
    }

    def __init__(self, params:GenerationParameters, device):
        cls = self.getPipelineClass(params)
        super().__init__(params=params, device=device, cls=cls)

    def getPipelineClass(self, params:GenerationParameters):
        self.features = self.getPipelineFeatures(params)
        return self.PIPELINE_MAP[(self.features.img2img, self.features.inpaint)]
