import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from PWR import utils
from PWR.prune import putils
from tensorboardX import SummaryWriter
import tqdm
from tqdm.auto import trange
import logging
import copy

log = logging.getLogger("Trainer")


class TrainHandler:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.writer = SummaryWriter("./")
        self.current_epoch = 0
        self.best_acc = 0.5
        self.basic_dict = utils.get_basic_dict(cfg)
        self.save_checkpoint("init")
        self.module_dict = utils.get_module_dict(self.basic_dict["model"])
        self.module_init()
        if self.cfg["model"]["pretrained"]:
            self.load_checkpoint(self.cfg["model"]["load_name"], self.cfg["model"]["load_path"])

        if cfg.seed == -1:
            cfg.seed = torch.initial_seed()
            log.info(f"Seed initialized: {cfg.seed}")
        else:
            torch.manual_seed(cfg.seed)
            log.info(f"Seed initialized with setting: {torch.initial_seed()}")

        # update config file and save it
        cfg = utils.get_cfg_data(cfg)
        OmegaConf.save(cfg, "config.yaml")
        log.info(OmegaConf.to_yaml(cfg))
        log.info("Trainer Initialized")

    def train_for(self, epoch):
        self.train(
            self.basic_dict["model"],
            self.basic_dict["criterion"],
            self.basic_dict["optimizer"],
            self.basic_dict["train_loader"],
            epoch,
            self.basic_dict["device"],
            self.basic_dict["scheduler"],
        )

    def train_to(self, epoch):
        self.train(
            self.basic_dict["model"],
            self.basic_dict["criterion"],
            self.basic_dict["optimizer"],
            self.basic_dict["train_loader"],
            epoch - self.current_epoch,
            self.basic_dict["device"],
            self.basic_dict["scheduler"],
        )

    def eval_train(self):
        return self.eval(
            self.basic_dict["model"],
            self.basic_dict["criterion"],
            self.basic_dict["train_loader"],
            self.basic_dict["device"],
        )

    def eval_test(self):
        return self.eval(
            self.basic_dict["model"],
            self.basic_dict["criterion"],
            self.basic_dict["test_loader"],
            self.basic_dict["device"],
        )

    def train(
        self, model, criterion, optimizer, data_loader, epochs, device, scheduler=None
    ):
        loop = trange(
            epochs,
            total=epochs,
            position=1,
            leave=False,
            file=sys.stdout,
            dynamic_ncols=True,
        )
        for _ in loop:
            loss_tmp = self.train_once(model, criterion, optimizer, data_loader, device)
            if loss_tmp > 1e4:
                break
            loss_train, acc_train = self.eval_train()
            loss_test, acc_test = self.eval_test()
            self.current_epoch += 1
            self.writer.add_scalar("loss/train", loss_train, self.current_epoch)
            self.writer.add_scalar("loss/test", loss_test, self.current_epoch)
            self.writer.add_scalar("acc/train", acc_train, self.current_epoch)
            self.writer.add_scalar("acc/test", acc_test, self.current_epoch)
            log.info(
                f"Epoch {self.current_epoch-1:3d} "
                f"TrainAcc: {acc_train*100:2.2f}% "
                f"TestAcc: {acc_test*100:2.2f}% "
                f"TrainLoss: {loss_train:2.4f} "
                f"TestLoss: {loss_test:2.4f} "
            )
            if acc_test > self.best_acc:
                self.save_checkpoint("best")
                self.best_acc = acc_test
            if scheduler is not None:
                scheduler.step()
        loop.close()

    def train_once(self, model, criterion, optimizer, data_loader, device):
        model.train()
        loop = tqdm.tqdm(
            enumerate(data_loader),
            position=0,
            leave=False,
            file=sys.stdout,
            total=len(data_loader),
            dynamic_ncols=True,
        )
        loop.set_description(f"Training Epoch {self.current_epoch}")
        for batch_idx, (data, target) in loop:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            if loss.item() > 1e4:
                return loss.item()
            loss.backward()
            optimizer.step()
            idx = self.current_epoch * len(data_loader) + batch_idx
            self.writer.add_scalar("loss/batch", loss, idx)
            if self.cfg["plot_in_out"]:
                self.plot_input_and_output(data, output, idx)
            loop.set_postfix({"loss": f"{loss.item():2.6f}"})
        if self.cfg["plot_module"]:
            self.plot_module_dict(self.module_dict, self.current_epoch)
        loop.close()
        return loss.item()

    @torch.no_grad()
    def eval(self, model, criterion, data_loader, device):
        model.eval()
        loss = 0
        correct = 0
        count = len(data_loader)
        size = len(data_loader.dataset)
        loop = tqdm.tqdm(
            enumerate(data_loader),
            position=0,
            leave=False,
            file=sys.stdout,
            total=count,
            dynamic_ncols=True,
        )
        loop.set_description(f"Evaluation Epoch {self.current_epoch}")
        for batch_idx, (data, target) in loop:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        return loss / count, correct / size

    def get_example_data(self):
        example_inputs, _ = next(iter(self.basic_dict["train_loader"]))
        example_inputs = example_inputs[0, :, :, :].unsqueeze(0)
        return example_inputs

    def get_current_state(self, mos=None):
        if mos is None:
            return
        state = {}
        if mos[0]:
            state["model"] = self.basic_dict["model"].state_dict()
        if mos[1]:
            state["optimizer"] = self.basic_dict["optimizer"].state_dict()
        if mos[2] and (self.basic_dict["scheduler"] is not None):
            state["scheduler"] = self.basic_dict["scheduler"].state_dict()
        return copy.deepcopy(state)

    def load_state(self, state, mos=None):
        if mos is None:
            return
        if mos[0]:
            self.basic_dict["model"].load_state_dict(state["model"])
        if mos[1]:
            self.basic_dict["optimizer"].load_state_dict(state["optimizer"])
        if mos[2] and (self.basic_dict["scheduler"] is not None):
            self.basic_dict["scheduler"].load_state_dict(state["scheduler"])

    def save_checkpoint(self, name, path=None):
        if path is None:
            path = "./saved"
        state_dict = {
            "epoch": self.current_epoch,
            "model": self.basic_dict["model"].state_dict(),
            "optimizer": self.basic_dict["optimizer"].state_dict(),
        }
        os.system(f"mkdir -p saved")
        torch.save(state_dict, f"{path}/{name}.pth")

    def load_checkpoint(self, name, path=None):
        if path is None:
            path = "./saved"
        state_dict = torch.load(f"{path}/{name}.pth")
        # self.current_epoch = state_dict["epoch"]
        # TODO: when load a pruned model
        try:
            self.basic_dict["model"].load_state_dict(state_dict["model"])
        except:
            log.info("Try to load model with pruning")
            putils.apply_fooweight(self.module_dict)
            self.basic_dict["model"].load_state_dict(state_dict["model"])
        self.basic_dict["model"].to(self.basic_dict["device"])

    def plot_module_dict(self, module_dict, idx):
        mean_dict = {}
        var_dict = {}
        for name, module in module_dict.items():
            mean_dict[name] = module.weight.mean()
            var_dict[name] = module.weight.var()
            self.writer.add_histogram(
                f"train/weight_histogram/{name}", module.weight, idx
            )
        self.writer.add_scalars("train/weight_mean", mean_dict, idx)
        self.writer.add_scalars("train/weight_var", var_dict, idx)

    def plot_input_and_output(self, data, output, idx):
        mean_dict = {}
        var_dict = {}
        mean_dict["input"] = data.mean()
        var_dict["input"] = data.var()
        mean_dict["output"] = output.mean()
        var_dict["output"] = output.var()
        self.writer.add_scalars("data/input_mean", mean_dict, idx)
        self.writer.add_scalars("data/output_var", var_dict, idx)
        self.writer.add_histogram("data/input_histogram", data, idx)
        self.writer.add_histogram("data/output_histogram", output, idx)

    def module_init(self):
        for m in self.basic_dict["model"].modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
