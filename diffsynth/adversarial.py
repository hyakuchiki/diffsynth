import torch
import torch.nn as nn
import torch.nn.functional as F

from diffsynth.layers import MLP
from diffsynth.model import EstimatorSynth
from diffsynth.spectral import compute_lsd, Mfcc
# domain adversarial training

class GradientReversalFunction(torch.autograd.Function):
    # https://cyberagent.ai/blog/research/11863/
    @staticmethod
    def forward(ctx, input_forward: torch.Tensor, scale: torch.Tensor):
        ctx.save_for_backward(scale)
        return input_forward
 
    @staticmethod
    def backward(ctx, grad_backward: torch.Tensor):
        scale, = ctx.saved_tensors
        return scale * -grad_backward, None

class GradientReversal(nn.Module):
    def __init__(self, scale: float = 1.0):
        super(GradientReversal, self).__init__()
        self.scale = torch.tensor(scale)
 
    def set_scale(self, scale):
        self.scale = torch.tensor(scale).to(self.scale.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.scale)

class AdversarialEstimatorSynth(EstimatorSynth):
    def __init__(self, estimator, synth, enc_dims, grl_scale=1.0) -> None:
        # estimator: not really the estimator but encodes feature
        super().__init__(estimator, synth)
        self.est_mlp = MLP(enc_dims, enc_dims)
        self.est_out = nn.Linear(enc_dims, synth.ext_param_size)
        # domain classifier
        self.cls_mlp = MLP(enc_dims, enc_dims)
        self.cls_out = nn.Linear(enc_dims, 1) # 2 domains
        self.grl = GradientReversal(scale=grl_scale)
    
    def estimate_param(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: estimated parameters in Tensor ranged 0~1
        """
        conditioning = self.estimator(conditioning)
        # not really the param but a feature
        z = torch.sigmoid(conditioning.pop('est_param'))
        conditioning['z'] = z
        est_param = torch.sigmoid(self.est_out(self.est_mlp(z)))
        return est_param, conditioning

    def set_grlscale(self, scale):
        self.grl.set_scale(scale)

    def get_logit(self, conditioning):
        # dont use estimator or synthesizer just get logits
        conditioning = self.estimator(conditioning)
        # not really the param but a feature
        z = torch.sigmoid(conditioning.pop('est_param'))
        logit = self.cls_out(self.cls_mlp(self.grl(z)))
        return logit

    def forward(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: audio
        """
        audio_length = conditioning['audio'].shape[1]
        est_param, conditioning = self.estimate_param(conditioning)
        # synthesize
        params_dict = self.synth.fill_params(est_param, conditioning)
        resyn_audio, outputs = self.synth(params_dict, audio_length)
        # classify
        logit = self.cls_out(self.cls_mlp(self.grl(conditioning['z'])))
        outputs['logit'] = logit
        return resyn_audio, outputs
    
    def domain_loss(self, logit, domain_label):
        # batch, n_frames, 1
        n_frames = logit.shape[1]
        domain_label = domain_label.unsqueeze(1).expand(-1, n_frames, -1)
        return F.binary_cross_entropy_with_logits(logit, domain_label)

    def losses(self, target, output, **loss_args):
        #default values
        args = {'param_w': 1.0, 'sw_w':1.0, 'enc_w':1.0, 'mfcc_w':1.0, 'lsd_w': 1.0, 'cls_w': 1.0, 'acc_w': 1.0, 'sw_loss': None, 'ae_model': None}
        args.update(loss_args)
        loss = self.audio_losses(target['audio'], output['output'], **args)
        if args['param_w'] > 0.0 and 'params' in target:
            loss['param'] = args['param_w'] * self.param_loss(output, target['params'])
        else:
            loss['param'] = 0.0
        if args['cls_w']>0.0:
            loss['cls'] = args['cls_w'] * self.domain_loss(output['logit'], target['domain'])
        else:
            loss['cls'] = 0
        if args['acc_w']>0.0: # only for eval
            n_frames = output['logit'].shape[1]
            domain_label = target['domain'].unsqueeze(1).expand(-1, n_frames, -1)
            loss['acc'] = (torch.sigmoid(output['logit'].detach()).round() == domain_label).sum()/output['logit'].numel()
        else:
            loss['acc'] = 0
        return loss
    
    def train_adversarial(self, syn_loader, real_loader, optimizer, device, loss_weights, sw_loss=None, ae_model=None, clip=1.0):
        self.train()
        sum_loss = 0
        loss_args = loss_weights.copy()
        loss_args['ae_model'] = ae_model
        loss_args['sw_loss'] = sw_loss
        assert len(syn_loader) == len(real_loader)
        for syn_dict, real_dict in zip(syn_loader, real_loader):
            # send data to device
            params = syn_dict.pop('params')
            params = {name:tensor.to(device, non_blocking=True) for name, tensor in params.items()}
            syn_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in syn_dict.items()}
            syn_dict['params'] = params
            if loss_args['sw_w']+loss_args['enc_w']+loss_args['mfcc_w']+loss_args['lsd_w'] == 0:
                # do not render audio because reconstruction is unnecessary
                synth_params, outputs = self.get_params(syn_dict)
                logit = self.cls_out(self.cls_mlp(self.grl(outputs['z'])))
                cls_loss = self.domain_loss(logit, syn_dict['domain'])
                batch_loss = loss_args['param_w']*self.param_loss(synth_params, syn_dict['params']) + loss_args['cls_w']*cls_loss
            else:
                # render audio
                resyn_audio, outputs = self(syn_dict)
                # Parameter loss
                losses = self.losses(syn_dict, outputs, **loss_args)
                batch_loss = sum(losses.values())
            
            # only classification loss for real data
            if loss_args['cls_w'] > 0.0:
                real_dict = {name:tensor.to(device, non_blocking=True) for name, tensor in real_dict.items()}
                logit = self.get_logit(real_dict)
                cls_loss = self.domain_loss(logit, real_dict['domain'])
                batch_loss += loss_args['cls_w']*cls_loss

            # Perform backward
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            sum_loss += batch_loss.detach().item()
        sum_loss /= len(syn_loader)
        return sum_loss