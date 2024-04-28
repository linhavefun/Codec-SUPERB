import torch
from transformers import AutoModel, AutoProcessor
from SoundCodec.base_codec.general import save_audio, ExtractedUnit
from audiocraft.utils import export
from audiocraft.models import CompressionModel
# export.export_pretrained_compression_model('facebook/encodec_32khz', '/data/zeyuet/project/trailer_music/audiocraft/output/audiocraft_zeyuet/output_model/compression_state_dict.bin')
export.export_encodec(
    '/aifs4su/mmcode/codeclm/encodec/audiocraft/output/audiocraft_ruibiny/xps/f6b25820/checkpoint.th',
    '/aifs4su/mmcode/codeclm/encodec/audiocraft/output/audiocraft_ruibiny/xps/f6b25820/compression_state_dict.bin')


class BaseCodec:
    def __init__(self):
        self.config()
        ckpt = '/aifs4su/mmcode/codeclm/encodec/audiocraft/output/audiocraft_ruibiny/xps/f6b25820/compression_state_dict.bin'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model = AutoModel.from_pretrained(self.pretrained_model_name).to(self.device)
        # self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name)
        # self.sampling_rate = self.processor.sampling_rate
        self.model =  CompressionModel.get_pretrained(ckpt)
        self.model.to(self.device)

    def config(self):
        self.pretrained_model_name = "encodec_32khz_retrain"

    @torch.no_grad()
    def synth(self, data, local_save=True):
        extracted_unit = self.extract_unit(data)
        data['unit'] = extracted_unit.unit
        audio_values = self.decode_unit(extracted_unit.stuff_for_synth)
        if local_save:
            audio_path = f"dummy_{self.pretrained_model_name}/{data['id']}.wav"
            save_audio(audio_values, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_values
        return data

    @torch.no_grad()
    def extract_unit(self, data):
        wav, sr = data["audio"]["array"], data["audio"]["sampling_rate"]
        # unsqueeze to [B, T], if no batch, B=1
        wav = torch.tensor(wav).unsqueeze(0)
        wav = wav.unsqueeze(0)
        wav = wav.to(torch.float32).to(self.device)
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.model.encode(wav)
        # codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [B, n_q, T]
        codes = encoded_frames[0]
        return ExtractedUnit(
            unit=codes,
            stuff_for_synth=(encoded_frames[0], data['audio']['array'].shape[0])
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        encoded_frames, original_shape = stuff_for_synth
        audio_values = \
            self.model.decode(encoded_frames)[0]
        # trim the audio to the same length as the input
        audio_values = audio_values[:, :original_shape].cpu()
        return audio_values