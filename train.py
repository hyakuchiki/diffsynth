import hydra
import pytorch_lightning as pl
from plot import AudioLogger
import warnings
from pytorch_lightning.callbacks import ModelCheckpoint
from diffsynth.model import EstimatorSynth

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):
    pl.seed_everything(0, workers=True)
    warnings.simplefilter('ignore', RuntimeWarning)
    model = EstimatorSynth(cfg.model)
    logger = pl.loggers.TensorBoardLogger("tb_logs", "", default_hp_metric=False, version='')
    hparams = {'data': cfg.data.train_type, 'schedule': cfg.model.l_sched.name, 'synth': cfg.model.synth_name}
    # dummy value
    logger.log_hyperparams(hparams, {'val_id/lsd': 40, 'val_ood/lsd': 40})
    # log audio examples
    checkpoint_callback = ModelCheckpoint(monitor="val_ood/lsd", save_top_k=1, filename="epoch_{epoch:03}_{val_ood/lsd:.2f}", save_last=True, auto_insert_metric_name=False)
    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval='step'), AudioLogger(), checkpoint_callback]
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    datamodule = hydra.utils.instantiate(cfg.data)
    # make model
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()
