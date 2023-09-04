from PWR import Trainer, Pruner

def main(cfg):
    train_handler = Trainer.TrainHandler(cfg)
    prune_handler = Pruner.PruneHandler(cfg["prune"], train_handler)
    prune_handler.run()
    train_handler.writer.close()