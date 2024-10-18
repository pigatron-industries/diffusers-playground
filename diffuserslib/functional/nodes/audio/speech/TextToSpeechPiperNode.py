from diffuserslib.functional.types import *
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *

from diffuserslib.ModelUtils import getFile



class TextToSpeechPiperNode(FunctionalNode):

    def __init__(self, 
                 prompt:StringFuncType = "",
                 model:StringFuncType = "rhasspy/piper-voices/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx",
                 speakerid:IntFuncType = 0,
                 length_scale:FloatFuncType|None = None,
                 noise_scale:FloatFuncType|None = None,
                 noise_w:FloatFuncType|None = None,
                 name:str="audio_piper"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.addParam("model", model, str)
        self.addParam("speakerid", speakerid, int)
        self.addParam("length_scale", length_scale, float)
        self.addParam("noise_scale", noise_scale, float)
        self.addParam("noise_w", noise_w, float)
        self.tts = None
        
        
    def process(self, prompt:str, model:str, speakerid:int, noise_scale:float, length_scale:float, noise_w:float):
        from piper.voice import PiperVoice
        if (self.tts is None):
            model_file = getFile(model)
            config_file = getFile(model + ".json")
            self.tts = PiperVoice.load(model_file, config_file)

        audio_array = self.synthesize(prompt, speaker_id = speakerid, noise_scale = noise_scale, length_scale = length_scale, noise_w = noise_w)

        # Normalize the audio array to the range [0, 1]
        audio_scaled = (audio_array.squeeze() - np.min(audio_array)) / (np.max(audio_array) - np.min(audio_array))
        return Audio(audio_scaled, self.tts.config.sample_rate)
    

    def synthesize(self, prompt:str, sentence_silence:float = 0.5, speaker_id:int|None = None, 
                   noise_scale:float|None = None, length_scale:float|None = None, noise_w:float|None = None):
        assert self.tts is not None, "TTS model not loaded"
        sentence_phonemes = self.tts.phonemize(prompt)

        num_silence_samples = int(sentence_silence * self.tts.config.sample_rate)
        silence_array = np.zeros((1, num_silence_samples), dtype=np.float32)
        audio_array = np.zeros((1, 0), dtype=np.float32)

        if noise_scale is None:
            noise_scale = self.tts.config.noise_scale
        if length_scale is None:
            length_scale = self.tts.config.length_scale
        if noise_w is None:
            noise_w = self.tts.config.noise_w

        print(f"{self.tts.config.num_speakers} speakers available.")

        for phonemes in sentence_phonemes:
            phoneme_ids = self.tts.phonemes_to_ids(phonemes)
            phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
            phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
            scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)

            if (self.tts.config.num_speakers > 1) and (speaker_id is None):
                speaker_id = 0
            if (self.tts.config.num_speakers == 1):
                speaker_id = None

            sid = None
            if speaker_id is not None:
                sid = np.array([speaker_id], dtype=np.int64)

            # Synthesize through Onnx
            audio = self.tts.session.run(
                None,
                {
                    "input": phoneme_ids_array,
                    "input_lengths": phoneme_ids_lengths,
                    "scales": scales,
                    "sid": sid,
                },
            )[0].squeeze((0, 1))

            audio_array = np.concatenate((audio_array, audio, silence_array), axis=1)
        
        return audio_array