from matplotlib.pyplot import hist
from diffuserslib.functional.types import *
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.GlobalConfig import GlobalConfig
from transformers import BarkModel, BarkProcessor
from encodec import EncodecModel
from encodec.utils import convert_audio
import librosa
import torch
import huggingface_hub
import numpy as np

from .CustomHubert import CustomHubert
from .CustomTokenizer import CustomTokenizer

import bark

class AudioGenerationBarkNode(FunctionalNode):

    def __init__(self, 
                 sample:AudiosFuncType,
                 prompt:StringFuncType = "",
                 name:str="audio_bark"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.addParam("sample", sample, List[Audio])
        self.hubert_model = None
        self.tokenizer = None
        self.encodec = None
        self.processor = None
        self.bark_model = None
        
        
    def process(self, prompt:str, sample:List[Audio]):
        voice_preset = None
        if(sample is not None and len(sample) > 0):
            voice_preset = self.clone(sample[0])
        audio_array = self.generate(prompt, voice_preset)
        return Audio(audio_array, bark.SAMPLE_RATE)


    def clone(self, sample:Audio):
        if (self.hubert_model is None):
            path = huggingface_hub.hf_hub_download(repo_id = 'Dongchao/UniAudio', subfolder = None, filename = 'hubert_base_ls960.pt')
            self.hubert_model = CustomHubert(checkpoint_path=path, device = 'cpu')
        if (self.tokenizer is None):
            path = huggingface_hub.hf_hub_download(repo_id = 'GitMylo/bark-voice-cloning', subfolder = None, filename = 'quantifier_hubert_base_ls960_14.pth')
            self.tokenizer = CustomTokenizer.load_from_checkpoint(path = path, device = 'cpu')
        if (self.encodec is None):
            self.encodec = EncodecModel.encodec_model_24khz()
            self.encodec.set_target_bandwidth(6.0)
            self.encodec.to('cpu')

        audio_array = self.prepare_audio(sample).to('cpu')
        semantic_vectors = self.hubert_model.forward(audio_array, input_sample_hz = self.encodec.sample_rate)
        semantic_tokens = self.tokenizer.get_token(semantic_vectors).cpu()

        wav = convert_audio(audio_array, self.encodec.sample_rate, self.encodec.sample_rate, 1).unsqueeze(0).to('cpu')
        with torch.no_grad():
            encoded_frames = self.encodec.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze().cpu()
        
        # temp_file = tempfile.NamedTemporaryFile(suffix = ".npz", delete = True)
        # np.savez(temp_file,
        #     semantic_prompt=semantic_tokens,
        #     fine_prompt=codes,
        #     coarse_prompt=codes[:2, :]
        # )
        return { 
                "semantic_prompt": semantic_tokens.numpy(),
                "coarse_prompt": codes[:2, :].numpy(),
                "fine_prompt": codes.numpy() 
            }


    def generate(self, prompt:str, voice_preset = None):
        bark.preload_models()
        audio_array = bark.generate_audio(text=prompt, history_prompt=voice_preset)
        # Using Transformers:
        # if (self.processor is None or self.bark_model is None):
        #     self.processor = BarkProcessor.from_pretrained("suno/bark")
        #     self.bark_model = BarkModel.from_pretrained("suno/bark")
        # inputs = self.processor(prompt, voice_preset = voice_preset)
        # audio_array = self.bark_model.generate(**inputs)
        # audio_array = audio_array.detach().numpy().squeeze()s
        return audio_array


    def prepare_audio(self, audio:Audio):
        assert self.encodec is not None, "Encodec model is not initialized"
        audio_array = audio.audio_array
        target_sample_rate = self.encodec.sample_rate
        if(audio.sample_rate != target_sample_rate):
            audio_array = librosa.resample(audio.audio_array, orig_sr = audio.sample_rate, target_sr = target_sample_rate)
        if(audio_array.ndim > 1):
            audio_array = audio_array.mean(axis=0)
        return torch.FloatTensor(audio_array).unsqueeze(0)
    

