import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.food101n

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.ggrp


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if getattr(args, "smoke_test", False):
        cfg.TRAINER.GGRP.SMOKE_TEST = True


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.GGRP = CN()
    cfg.TRAINER.GGRP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.GGRP.CSC = False  # class-specific context
    cfg.TRAINER.GGRP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.GGRP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.GGRP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    cfg.DATASET.NUM_SHOTS = 16
    # Config for noise 
    cfg.DATASET.NOISE_LABEL = True
    cfg.DATASET.NOISE_RATE = 0.5
    cfg.DATASET.NOISE_TYPE = 'sym'
    cfg.DATASET.num_class = 100

    #config for ot
    cfg.DATASET.USE_OT = True
    cfg.DATASET.REG_FEAT = 1.0
    cfg.DATASET.REG_LAB = 1.0
    cfg.DATASET.CURRICLUM_EPOCH = 0
    cfg.DATASET.BEGIN_RATE = 0.3
    cfg.DATASET.CURRICLUM_MODE = 'linear'
    cfg.DATASET.PMODE = 'logP'
    cfg.DATASET.REG_E = 0.01

    # PromptOT 训练链路的可控开关
    cfg.TRAINER.GGRP.ENABLE_PROMPT_OT = True
    cfg.TRAINER.GGRP.AUDIT = True
    cfg.TRAINER.GGRP.AUDIT_TOL = 1e-2
    cfg.TRAINER.GGRP.AUDIT_ONLY_FIRST = True
    cfg.TRAINER.GGRP.SMOKE_TEST = False
    cfg.TRAINER.GGRP.SMOKE_MAX_EPOCH = 1
    cfg.TRAINER.GGRP.SMOKE_STEPS = 20
    # 置信度阈值（改用 margin，单位为 top1-top2 概率差），高门槛偏向保留 noisy
    cfg.TRAINER.GGRP.TEACHER_CONF_HIGH = 0.12
    cfg.TRAINER.GGRP.TEACHER_CONF_LOW = 0.05
    cfg.TRAINER.GGRP.TEACHER_EMA = 0.0
    # CE 目标：默认对齐教师伪标签，在高噪声场景下更稳
    cfg.TRAINER.GGRP.CE_TARGET = "teacher"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/DATA/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="output/caltech101_sym0.5_16shot", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/GGRP/vit_b16_ep100.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/caltech101.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="GGRP", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run a tiny smoke test to verify PromptOT teacher-student chain",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
