import torch
import torchcrepe
import torch.nn.functional as F
from torchaudio.models import wav2vec2_base, wav2vec2_large, wav2vec2_large_lv60k
from .perceptual import Perceptual

class Wav2VecLoss(Perceptual):
    """
    Use feature extractor only
    """
    def __init__(self, model_spec, state_dict):
        super().__init__()
        if model_spec == 'base':
            entire = wav2vec2_base(num_out=32)
        elif model_spec == 'large':
            entire = wav2vec2_large(num_out=32)
        elif model_spec == 'large_lv60k':
            entire = wav2vec2_large_lv60k(num_out=32)
        model = entire.feature_extractor
        model.load_state_dict(torch.load(state_dict), strict=False)
        self.model = model.eval()

    def perceptual_loss(self, target_audio, input_audio):
        # length = torch.ones(target_audio.shape[0], device=target_audio.device) * target_audio.shape[-1]
        target_embed = self.model(target_audio, None)[0]
        input_embed = self.model(input_audio, None)[0]
        return F.l1_loss(target_embed, input_embed)