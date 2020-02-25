import os
from pprint import pformat
from time import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from util.config import init_class
from util.meters import AverageMeter

from . import BaseAgent


class ClassificationAgent(BaseAgent):
    """Agent for PhysioNet 2020 challenge classification experiments.

    Experiment Configuration keys:
    - seed (int|bool): deterministic pytorch training
    - cpu_only (bool): if true, do not use cuda/CuDNN
    - dataset (dict): dataset configuration
    - val_split (float): ratio of dataset to split for validation
    - train_loader (dict): training dataloader
    - val_loader (dict): validation dataloader
    - model (dict): model architecture
    - criterion (dict): loss function
    - optimizer (dict): training optimizer
    - scheduler (dict): learning rate scheduler
    - epochs (int): number of epochs to train
    - log_per_num_batch (int): number of minibatches per log
    - log_num_activation_samples (int): number of samples per batch to log activations for
    - validate_zero (bool): if true, run a validation epoch pre-training
    - checkpoint (str): filename for most recent checkpoint
    - best_checkpoint (str): filename for best checkpoint
    """

    def __init__(self, config):
        super(ClassificationAgent, self).__init__(config)

        # Tensorboard Summary Writer
        self.tb_sw = SummaryWriter(log_dir=config["tb_dir"])
        self.tb_sw.add_text("config", pformat(config))

        self._set_seed(config)
        self.tb_sw.add_text("seed", str(self.seed))

        # Setup CUDA
        self._setup_cuda(config)

        # Initialize dataset and dataloaders
        ds = init_class(config.get("dataset"))
        val_len = int(len(ds) * config.get("val_split"))
        train_len = len(ds) - val_len
        self.train_set, self.val_set = torch.utils.data.random_split(
            ds, (train_len, val_len,)
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            **config.get("train_loader"),
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_set,
            **config.get("val_loader"),
        )
        # self.train_set, self.train_loader = BaseAgent._setup_data(config.get("train"))
        # self.val_set, self.val_loader = BaseAgent._setup_data(config.get("val"))

        # Initialize the neural network model architecture
        self.model = init_class(config.get("model"))
        if self.use_cuda:
            self.model.cuda()
        try:
            # Attempt tensorboard model graph visualization
            vis_inp, _ = next(iter(self.val_set))
            self.tb_sw.add_graph(self.model, vis_inp.unsqueeze(0))
        except Exception as e:
            self.logger.warn(e)

        # Initialize task loss criterion, optimizer, scheduler
        self.criterion = init_class(config.get("criterion"))
        self.optimizer = init_class(config.get("optimizer"), self.model.parameters())
        sched_config = config.get("scheduler")
        if sched_config:
            self.logger.info("Scheduler: %s", pformat(sched_config))
            self.scheduler = init_class(config.get("scheduler"), self.optimizer)
        else:
            # this is a no-op scheduler that does not change the LR
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1, gamma=1
            )

        if self.use_cuda and len(self.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()

        # number of epochs to run
        self.epochs = config.get("epochs", 123)
        # checkpoint file path names
        self.checkpoint = config.get("checkpoint", "checkpoint.pth.tar")
        self.best_checkpoint = config.get("best_checkpoint", "model_best.pth.tar")

        self.epoch = 0
        self.best_acc1 = 0

        # log every set number of batches
        self.log_per_num_batch = config.get("log_per_num_batch", 1)
        # should a validation pass be done before training begins?
        self.validate_zero = config.get("validate_zero", False)

        self.logger.info("Train Dataset: %s", self.train_set)
        self.logger.info("Train batches per epoch: %d", len(self.train_loader))
        self.logger.info("Validation Dataset: %s", self.val_set)
        self.logger.info("Validation batches per epoch: %d", len(self.val_loader))
        self.logger.info("Loss Criterion: %s", self.criterion)
        self.logger.info("Optimizer: %s", self.optimizer)

        if config.get("resume"):
            resume_path = config.get("resume")
            if os.path.isfile(resume_path):
                self.logger.info("Loading checkpoint: %s", resume_path)
                checkpoint = torch.load(resume_path)
                self.epoch = checkpoint["epoch"]
                self.model.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])
                self.best_acc1 = checkpoint["best_acc1"]
            else:
                self.logger.info("No checkpoint found at '%s'", resume_path)

        self.logger.info(
            "Training from Epoch %(start)d to %(end)d",
            {"start": self.epoch, "end": self.epochs},
        )

    def run(self):
        self.exp_start = time()
        if self.epoch == 0 and self.validate_zero:
            with torch.no_grad():
                self.run_epoch_pass(self.val_loader, epoch=0, train=False)

        for epoch in range(self.epoch, self.epochs):
            epoch_data = dict(
                [
                    ("lr_{}".format(idx), lr)
                    for (idx, lr) in enumerate(self.scheduler.get_lr())
                ]
            )
            train_res = self.run_epoch_pass(self.train_loader, epoch=epoch)
            epoch_data.update(train_res)
            with torch.no_grad():
                val_res = self.run_epoch_pass(self.val_loader, epoch=epoch, train=False)
            epoch_data.update(val_res)
            self.scheduler.step()
            self.tb_sw.add_scalars("Epoch", epoch_data, global_step=epoch)

            is_best = val_res["Val_top1"] > self.best_acc1
            self.best_acc1 = max(val_res["Val_top1"], self.best_acc1)
            BaseAgent.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "best_acc1": self.best_acc1,
                },
                is_best,
                filename=os.path.join(self.config["chkpt_dir"], self.checkpoint),
                best_filename=os.path.join(
                    self.config["chkpt_dir"], self.best_checkpoint
                ),
            )

    def run_epoch_pass(self, dataloader, epoch=0, train=True):
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")

        if train:
            self.model.train()
            tag = "Train"
        else:
            self.model.eval()
            tag = "Val"
        t_desc = "{} Epoch {}/{}".format(tag, epoch, self.epochs)

        end = time()
        with tqdm(dataloader, total=len(dataloader), desc=t_desc) as t:
            for i, (signal, target) in enumerate(t):
                step = (epoch * len(dataloader)) + i

                # measure data loading time
                data_time.update(time() - end)

                if self.use_cuda:
                    signal = signal.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                # compute prediction
                outputs = self.model(signal)
                loss = self.criterion(outputs, target)

                if train:
                    # compute the gradient and to an optimizer step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # measure the accuracy and record the loss
                losses.update(loss.item(), outputs.size(0))

                # print(target.detach())
                # print(outputs.detach())
                # acc1 = accuracy_score(target.detach(), outputs.detach())
                # top1.update(acc1, outputs.size(0))

                # measure elapsed times
                batch_time.update(time() - end)

                # logging
                self.log_post_batch(step, t, losses, top1, tag)

                end = time()

        self.logger.info(
            "{} Epoch {}/{} ".format(tag, epoch, self.epochs)
            + "Loss {:.4f} ({:.4f}) | ".format(losses.val, losses.avg,)
            + "Top1 {:.1f} ({:.1f})".format(top1.val, top1.avg,)
        )

        return {
            f"{tag}_loss": losses.avg,
            f"{tag}_top1": top1.avg,
        }

    def log_post_batch(self, step, t, losses, top1, tag):
        # batch has just finished, log the necessary values
        if step % self.log_per_num_batch == 0:
            t.set_postfix_str(
                "Loss {:.4f} ({:.4f}) | ".format(losses.val, losses.avg,)
                + "Top1 {:.1f} ({:.1f})".format(top1.val, top1.avg,)
            )
            tag_scalar_dict = {
                "loss": losses.val,  # "loss_avg": losses.avg,
                "top1": top1.val,  # "top1_avg": top1.avg,
                # "data_time": data_time.val,  # "data_time_avg": data_time.avg,
                # "batch_time": batch_time.val,  # "batch_time_avg": batch_time.avg
            }
            self.tb_sw.add_scalars(tag, tag_scalar_dict, global_step=step)
