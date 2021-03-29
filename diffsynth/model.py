import torch.nn as nn

class EstimatorSynth(nn.Module):
    """
    audio -> Estimator -> Synth -> audio
    """
    def __init__(self, estimator, synth):
        super().__init__()
        self.estimator = estimator
        self.synth = synth

    def forward(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: audio
        """
        audio_length = conditioning['audio'].shape[1]
        conditioning = self.estimator(conditioning)
        est_param = conditioning['est_param']
        params = self.synth.fill_params(est_param, conditioning)
        resyn_audio, outputs = self.synth(params, audio_length)
        return resyn_audio