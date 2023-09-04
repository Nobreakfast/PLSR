from PWR import Trainer

def main(cfg):
    handler = Trainer.TrainHandler(cfg)
    handler.train_for(cfg.epoch)
    handler.writer.close()