from SoundCodec.base_codec.encodec_retrain import BaseCodec

class Codec(BaseCodec):
    def config(self):
        self.pretrained_model_name = "encodec_32khz_retrain"
        self.sampling_rate = 32_000
