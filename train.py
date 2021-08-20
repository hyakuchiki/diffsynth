import hydra
import pytorch_lightning as pl
import pprint
from plot import AudioLogger
import warnings

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):
    pl.seed_everything(0, workers=True)
    warnings.simplefilter('ignore', RuntimeWarning)
    model = hydra.utils.instantiate(cfg.model)
    logger = pl.loggers.TensorBoardLogger("tb_logs", "")
    # log audio examples
    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval='step'), AudioLogger()]
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    datamodule = hydra.utils.instantiate(cfg.data)
    # make model
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()
