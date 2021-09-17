import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import diffsynth.util as util
from diffsynth.spectral import compute_lsd, loudness_loss, Mfcc
import pytorch_lightning as pl
from diffsynth.modelutils import construct_synth_from_conf
from diffsynth.schedules import ParamSchedule
import hydra

class EstimatorSynth(pl.LightningModule):
    """
    audio -> Estimator -> Synth -> audio
    """
    def __init__(self, model_cfg, synth_cfg, sched_cfg):
        super().__init__()
        self.synth = construct_synth_from_conf(synth_cfg)
        self.estimator = hydra.utils.instantiate(model_cfg.estimator, output_dim=self.synth.ext_param_size)
        self.loss_w_sched = ParamSchedule(sched_cfg) # loss weighting
        self.sw_loss = hydra.utils.instantiate(model_cfg.sw_loss) # reconstruction loss
        if model_cfg.perc_model is not None:
            self.perc_model = hydra.utils.instantiate(model_cfg.perc_model)
        else:
            self.perc_model = None
        self.log_grad = model_cfg.log_grad
        self.lr = model_cfg.lr
        self.decay_rate = model_cfg.decay_rate
        self.mfcc = Mfcc(n_fft=1024, hop_length=256, n_mels=40, n_mfcc=20, sample_rate=16000)
        self.save_hyperparameters()

    def param_loss(self, synth_output, param_dict):
        loss = 0
        for k, target in param_dict.items():
            output_name = self.synth.dag_summary[k]
            x = synth_output[output_name]
            if target.shape[1] > 1:
                x = util.resample_frames(x, target.shape[1])
            loss += F.l1_loss(x, target)
        loss = loss / len(param_dict.keys())
        return loss

    def estimate_param(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: estimated parameters in Tensor ranged 0~1
        """
        return self.estimator(conditioning['audio'])

    def log_param_grad(self, params_dict):
        def save_grad(name):
            def hook(grad):
                # batch, n_frames, feat_size
                grad_v = grad.abs().mean(dim=(0, 1))
                for i, gv in enumerate(grad_v):
                    self.log('train/param_grad/'+name+f'_{i}', gv, on_step=False, on_epoch=True)
            return hook

        if self.log_grad:
            for k, v in params_dict.items():
                if v.requires_grad == True:
                    v.register_hook(save_grad(k))

    def forward(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: audio
        """
        audio_length = conditioning['audio'].shape[1]
        est_param = self.estimate_param(conditioning)
        params_dict = self.synth.fill_params(est_param, conditioning)
        if self.log_grad is not None:
            self.log_param_grad(params_dict)

        resyn_audio, outputs = self.synth(params_dict, audio_length)
        return resyn_audio, outputs

    def get_params(self, conditioning):
        """
        Don't render audio
        """
        est_param = self.estimate_param(conditioning)
        params_dict = self.synth.fill_params(est_param, conditioning)
        if self.log_grad is not None:
            self.log_param_grad(params_dict)
        
        synth_params = self.synth.calculate_params(params_dict)
        return synth_params

    def train_losses(self, target, output, loss_w=None, sw_loss=None, perc_model=None):
        sw_loss = self.sw_loss if sw_loss is None else sw_loss
        perc_model = self.perc_model if perc_model is None else perc_model
        # always computes mean across batch dimension
        if loss_w is None:
            loss_w = {'param_w': 1.0, 'sw_w':1.0, 'perc_w':1.0}
        loss_dict = {}
        # parameter L1 loss
        if loss_w['param_w'] > 0.0 and 'params' in target:
            loss_dict['param'] = loss_w['param_w'] * self.param_loss(output, target['params'])
        else:
            loss_dict['param'] = 0.0
        # Audio losses
        target_audio = target['audio']
        resyn_audio = output['output']
        if loss_w['sw_w'] > 0.0 and sw_loss is not None:
            # Reconstruction loss
            spec_loss, wave_loss = sw_loss(target_audio, resyn_audio)
            loss_dict['spec'], loss_dict['wave'] = loss_w['sw_w'] * spec_loss, loss_w['sw_w'] * wave_loss
        else:
            loss_dict['spec'], loss_dict['wave'] = (0, 0)
        if loss_w['perc_w'] > 0.0 and perc_model is not None:
            loss_dict['perc'] = loss_w['perc_w']*perc_model.perceptual_loss(target_audio, resyn_audio)
        else:
            loss_dict['perc'] = 0
        return loss_dict

    def monitor_losses(self, target, output):
        mon_losses = {}
        # Audio losses
        target_audio = target['audio']
        resyn_audio = output['output']
        # losses not used for training
        mon_losses['lsd'] = compute_lsd(target_audio, resyn_audio)
        mon_losses['loud'] = loudness_loss(resyn_audio, target_audio)
        mon_losses['mfcc'] = F.l1_loss(self.mfcc(target_audio), self.mfcc(resyn_audio))
        return mon_losses

    def training_step(self, batch_dict, batch_idx):
        # get loss weights
        loss_weights = self.loss_w_sched.get_parameters(self.global_step)
        self.log_dict({'lw/'+k: v for k, v in loss_weights.items()}, on_epoch=True, on_step=False)
        if loss_weights['sw_w']+loss_weights['perc_w'] == 0:
            # do not render audio because reconstruction is unnecessary
            synth_params = self.get_params(batch_dict)
            # Parameter loss
            batch_loss = loss_weights['param_w'] * self.param_loss(synth_params, batch_dict['params'])
        else:
            # render audio
            resyn_audio, outputs = self(batch_dict)
            losses = self.train_losses(batch_dict, outputs, loss_weights)
            self.log_dict({'train/'+k: v for k, v in losses.items()}, on_epoch=True, on_step=False)
            batch_loss = sum(losses.values())
        self.log('train/total', batch_loss, prog_bar=True, on_epoch=True, on_step=False)
        return batch_loss

    def validation_step(self, batch_dict, batch_idx, dataloader_idx):
        # render audio
        resyn_audio, outputs = self(batch_dict)
        losses = self.train_losses(batch_dict, outputs)
        eval_losses = self.monitor_losses(batch_dict, outputs)
        losses.update(eval_losses)
        prefix = 'val_id/' if dataloader_idx==0 else 'val_ood/'
        losses = {prefix+k: v for k, v in losses.items()}
        self.log_dict(losses, prog_bar=True, on_epoch=True, on_step=False, add_dataloader_idx=False)
        return losses

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("val_id/wave", None)
        return items

    def test_step(self, batch_dict, batch_idx, dataloader_idx):
        # render audio
        resyn_audio, outputs = self(batch_dict)
        losses = self.train_losses(batch_dict, outputs)
        eval_losses = self.monitor_losses(batch_dict, outputs)
        losses.update(eval_losses)
        prefix = 'val_id/' if dataloader_idx==0 else 'val_ood/'
        losses = {prefix+k: v for k, v in losses.items()}
        self.log_dict(losses, prog_bar=True, on_epoch=True, on_step=False, add_dataloader_idx=False)
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.estimator.parameters(), self.lr)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decay_rate)
            }
        }