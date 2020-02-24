#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import PhysioNet2020Dataset


def parse_args(procs):
    p = ArgumentParser(
        "PhysioNet 2020 Challenge Experiment Runner",
        formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "-i", "--in-dir",
        help="directory containing *.hea and *.mat files",
        default="Training_WFDB")
    p.add_argument(
        "-o", "--out-dir",
        default=os.path.join(os.getcwd(), "out"),
        help="Experiment output directory")
    p.add_argument(
        "-p", "--pool", default=procs, type=int,
        help="number of multiprocessing cores to use, set 0 for no pooling")
    p.add_argument(
        "--cpu-only", default=False, type=bool,
        help="if True, do not use CUDA/GPUs")
    p.add_argument(
        "--gpu-ids", default=list(range(torch.cuda.device_count())), type=int, nargs="+",
        help="GPU ids for multigpu training"
    )
    p.add_argument(
        "--seed", default=False, help="if True, enable deterministic training"
    )

    data_g = p.add_argument_group("Dataset")
    data_g.add_argument(
        "-b", "--train-batch-size", default=64, type=int,
        help="dataset training batch size"
    )
    data_g.add_argument(
        "--val-batch-size", default=200, type=int,
        help="dataset validation batch size"
    )
    data_g.add_argument(
        "--sample-rate", default=200, type=int,
        help="sample rate (frequency, hz) of signal"
    )
    data_g.add_argument(
        "--max-seq-len", default=4000, type=int,
        help="max length of each signal passed to model"
    )
    data_g.add_argument(
        "--val-split", default=0.1, type=float,
        help="portion of dataset to use for validation"
    )
    data_g.add_argument(
        "--num-train-workers", default=max(procs/3, 1), type=int,
        help="number of train dataloader workers"
    )
    data_g.add_argument(
        "--num-val-workers", default=max(procs/4, 1), type=int,
        help="number of validation dataloader workers"
    )

    return p.parse_args()


def main():
    try:
        # multiprocessing number of workers
        procs = len(os.sched_getaffinity(0))
    except:
        procs = 1

    args = parse_args(procs)

    sw_dir = os.path.join(args.out_dir, "tb")
    sw = SummaryWriter(sw_dir)

    # setup cuda
    cpu_only = args.cpu_only
    gpu_ids = args.gpu_ids
    use_cuda = torch.cuda.is_available() and not cpu_only and gpu_ids
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = " ".join(map(str, gpu_ids))
        for gpu_id in gpu_ids:
            device_name = torch.cuda.get_device_name(gpu_id)

    # setup seed
    seed = args.seed
    if seed is not False:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load and split the dataset
    dataset = PhysioNet2020Dataset(
        args.in_dir, fs=args.sample_rate, max_seq_len=args.max_seq_len, proc=args.pool)
    val_data_len = int(len(dataset) * args.val_split)
    train_data_len = len(dataset) - val_data_len
    train_set, val_set = random_split(dataset, (train_data_len, val_data_len,))

    # configure the dataloaders
    train_loader = DataLoader(
        train_set, batch_size=args.train_batch_size,
        shuffle=True, num_workers=args.num_train_workers,
        collate_fn=PhysioNet2020Dataset.collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=args.val_batch_size,
        num_workers=args.num_val_workers,
        collate_fn=PhysioNet2020Dataset.collate_fn
    )

    # initialize the model

    sw.close()


if __name__ == "__main__":
    main()
