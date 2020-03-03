import os
from pprint import pformat
from time import time

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from apex import amp
from datasets import PhysioNet2020Dataset
from util.config import init_class
from util.evaluation import compute_auc, compute_beta_score
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
    - use_amp (boolean): if true, nvidia/apex acceleration should be used
    - amp_opt_level (str): "O1" optimization default
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
        self.use_amp = config.get("use_amp", False)
        self.amp_opt_level = config.get("amp_opt_level", "O1")

        # Initialize dataset and dataloaders
        data_dir = config["dataset"]["kwargs"]["data_dir"]
        val_split = config.get("val_split", 0)
        train_records, val_records = PhysioNet2020Dataset.split_names(
            data_dir, 1 - val_split
        )
        self.logger.debug("Training using records: %s", train_records)
        self.logger.debug("Validating using records: %s", val_records)

        self.logger.info(
            "From records in %s, training on %d records, validating on %d records",
            data_dir,
            len(train_records),
            len(val_records),
        )
        self.train_set = init_class(config.get("dataset"), records=train_records)
        self.val_set = init_class(config.get("dataset"), records=val_records)

        collate_fn = None
        if config.get("use_collate_fn", False):
            collate_fn = PhysioNet2020Dataset.collate_fn

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, **config.get("train_loader"), collate_fn=collate_fn
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_set, **config.get("val_loader"), collate_fn=collate_fn
        )

        # Initialize the neural network model architecture
        self.model = init_class(config.get("model"))
        if self.use_cuda:
            self.model.cuda()
        try:
            # Attempt tensorboard model graph visualization
            vis_inp, _ = next(iter(self.val_set))
            if self.use_cuda:
                vis_inp = vis_inp.cuda()
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

        if self.use_cuda:
            if self.use_amp:
                amp.initialize(self.model, self.optimizer, opt_level=self.amp_opt_level)
            if len(self.gpu_ids) > 1:
                self.model = torch.nn.DataParallel(self.model)

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
                if self.use_amp:
                    amp.load_state_dict(checkpoint.get("amp"))
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
            self.scheduler.step()
            with torch.no_grad():
                val_res = self.run_epoch_pass(self.val_loader, epoch=epoch, train=False)
            epoch_data.update(val_res)
            self.tb_sw.add_scalars("Epoch", epoch_data, global_step=epoch)

            is_best = val_res["Val_acc"] > self.best_acc1
            self.best_acc1 = max(val_res["Val_acc"], self.best_acc1)
            checkpoint_data = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_acc1": self.best_acc1,
            }
            if self.use_amp:
                checkpoint_data["amp"] = amp.state_dict()
            BaseAgent.save_checkpoint(
                checkpoint_data,
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
        acc_meter = AverageMeter("Acc", ":6.2f")
        auroc_meter = AverageMeter("AUROC", ":.3f")
        auprc_meter = AverageMeter("AUPRC", ":.3f")
        f_measure_meter = AverageMeter("F-Measure", ":.3f")
        f_beta_meter = AverageMeter("Fbeta-Measure", ":.3f")
        g_beta_meter = AverageMeter("Gbeta-Measure", ":.3f")

        if train:
            self.model.train()
            tag = "Train"
        else:
            self.model.eval()
            tag = "Val"
        t_desc = "{} Epoch {}/{}".format(tag, epoch, self.epochs)

        end = time()
        with tqdm(dataloader, total=len(dataloader), desc=t_desc) as t:
            for i, batch in enumerate(t):
                step = (epoch * len(dataloader)) + i

                # measure data loading time
                data_time.update(time() - end)

                if self.use_cuda:
                    batch["signal"] = batch["signal"].cuda(non_blocking=True)
                    batch["target"] = batch["target"].cuda(non_blocking=True)

                # compute prediction
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch["target"])

                if train:
                    # compute the gradient and to an optimizer step
                    self.optimizer.zero_grad()
                    if self.use_amp:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    self.optimizer.step()

                # measure the accuracy and record the loss
                bs = outputs.size(0)
                losses.update(loss.item(), bs)

                (
                    (scaled_prediction, target_ground_truths),
                    (auroc, auprc, acc, f_measure, f_beta, g_beta),
                ) = self.calculate_evaluation(
                    outputs.detach(), batch["target"].detach(), tag=tag
                )

                acc_meter.update(acc, bs)
                auroc_meter.update(auroc, bs)
                auprc_meter.update(auprc, bs)
                f_measure_meter.update(f_measure, bs)
                f_beta_meter.update(f_beta, bs)
                g_beta_meter.update(g_beta, bs)

                # measure elapsed times
                batch_time.update(time() - end)

                # logging
                self.log_post_batch(
                    step,
                    t,
                    losses,
                    tag,
                    scaled_prediction,
                    target_ground_truths,
                    acc_meter,
                    auroc_meter,
                    auprc_meter,
                    f_measure_meter,
                    f_beta_meter,
                    g_beta_meter,
                )

                end = time()

        self.logger.info(
            "{} Epoch {}/{} ".format(tag, epoch, self.epochs)
            + "Loss {:.4f} ({:.4f}) | ".format(losses.val, losses.avg,)
            + "Top1 {:.3f} ({:.3f})".format(acc_meter.val, acc_meter.avg,)
        )

        return {
            f"{tag}_loss": losses.avg,
            f"{tag}_acc": acc_meter.avg,
        }

    def log_post_batch(
        self,
        step,
        t,
        losses,
        tag,
        scaled_prediction,
        target_ground_truths,
        acc_meter,
        auroc_meter,
        auprc_meter,
        f_measure_meter,
        f_beta_meter,
        g_beta_meter,
    ):
        # batch has just finished, log the necessary values
        if step % self.log_per_num_batch == 0:
            t.set_postfix_str(
                "Loss {:.4f} ({:.4f}) | ".format(losses.val, losses.avg,)
                + "Acc {:.3f} ({:.3f})".format(acc_meter.val, acc_meter.avg,)
            )
            tag_scalar_dict = {
                "loss": losses.val,  # "loss_avg": losses.avg,
                "acc": acc_meter.val,  # "top1_avg": top1.avg,
                "auroc": auroc_meter.val,
                "auprc": auprc_meter.val,
                "f_measure": f_measure_meter.val,
                "f_beta": f_beta_meter.val,
                "g_beta": g_beta_meter.val,
                # "data_time": data_time.val,  # "data_time_avg": data_time.avg,
                # "batch_time": batch_time.val,  # "batch_time_avg": batch_time.avg
            }
            self.tb_sw.add_scalars(tag, tag_scalar_dict, global_step=step)
            self.tb_sw.add_pr_curve(
                tag, target_ground_truths, scaled_prediction, global_step=step
            )

    def calculate_evaluation(self, predictions, targets, threshold=0.9, tag="Train"):
        scaler = MinMaxScaler()
        scaled_prediction = scaler.fit_transform(predictions.cpu().T).T
        output_predictions = scaled_prediction >= threshold

        target_ground_truths = targets.cpu().numpy() > 0

        acc, f_measure, f_beta, g_beta = compute_beta_score(
            target_ground_truths, output_predictions
        )
        auroc, auprc = compute_auc(target_ground_truths, scaled_prediction)

        return (
            (scaled_prediction, target_ground_truths),
            (auroc, auprc, acc, f_measure, f_beta, g_beta),
        )

    def finalize(self):
        """Called after `run` by main.py. Clean up experiment.
        """
        self.tb_sw.close()
