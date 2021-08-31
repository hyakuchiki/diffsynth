# Diffsynth - a Differentiable Musical Synthesizer in PyTorch

Synthesizer Sound Matching with Differentiable DSP @ ISMIR2021
https://hyakuchiki.github.io/DiffSynthISMIR/

## Features

- Additive-subtractive synthesizer
- FM synthesizer
- ADSR envelopes, LFOs
- Chorus/flanger, reverb effects
- Parameter estimator network

## To-do

- Training with perceptual loss doesn't work

## Training

- p-loss model
	- `python train.py experiment=only_param_h2of trainer.gpus=1`
- pretrain
	- `python train.py experiment=pretrain_h2of trainer.gpus=1`
- resume real model
	- `python train.py experiment=resume_real_h2of trainer.gpus=1  trainer.resume_from_checkpoint=[pretrain ckpt absolute path]`
- resume synth model
	- `python train.py experiment=resume_synth_h2of trainer.gpus=1  trainer.resume_from_checkpoint=[pretrain ckpt absolute path]`

## Notes

Several features have been added since ISMIR2021 (code is more readable, chorus effect, etc.)
To reproduce results from ISMIR2021, revert to [Ver. May2021](https://github.com/hyakuchiki/diffsynth/commit/aca9585a8c0f8466166830dfed97bf222d7e1f40)
