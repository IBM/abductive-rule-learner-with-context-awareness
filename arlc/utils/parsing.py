# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

import argparse

def eval_parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", type=str)
    arg_parser.add_argument("--seeds", type=int)
    args = arg_parser.parse_args()
    return args

def parse_args():
    arg_parser = argparse.ArgumentParser(description="NVSA lernable backend training and evaluation on RAVEN")
    arg_parser.add_argument("--run_name", type=str)
    arg_parser.add_argument("--mode", type=str, default="train", help="Train/test")
    arg_parser.add_argument("--exp_dir", type=str, default="results/")
    arg_parser.add_argument("--data_dir", type=str, default="dataset/")
    arg_parser.add_argument("--rule_type", type=str, default="arlc")
    arg_parser.add_argument("--num_terms", type=int, default=12)
    arg_parser.add_argument("--resume", type=str, default="", help="Resume from a initialized model")
    arg_parser.add_argument("--seed", type=int, default=1234, help="Random number seed")
    arg_parser.add_argument("--run", type=int, default=0, help="Run id")
    arg_parser.add_argument("--config", type=str, default="center_single", help="The configuration used for training")
    arg_parser.add_argument("--gen_attribute", type=str, default="", help="Generalization experiment [Type, Size, Color]")
    arg_parser.add_argument("--gen_rule", type=str, default="", help="Generalization experiment [Arithmetic, Constant, Progression, Distribute_Three]")
    arg_parser.add_argument("--n-train", type=int, default=None)
    arg_parser.add_argument(  "--model",  type=str,  default="LearnableFormula",  help="Model used in the reasoner (LearnableFormula, MLP)",  )
    arg_parser.add_argument(  "--epochs", type=int, default=50, help="The number of training epochs"  )
    arg_parser.add_argument("--batch_size", type=int, default=4, help="Size of batch")
    arg_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    arg_parser.add_argument(  "--weight-decay",  type=float,  default=0,  help="Weight decay of optimizer, same as l2 reg",  )
    arg_parser.add_argument(  "--num_workers", type=int, default=8, help="Number of workers for data loader"  )
    arg_parser.add_argument(  "--clip",  type=float,  default=10,  help="Max value/norm in gradient clipping (now l2 norm)",  )
    arg_parser.add_argument(  "--vsa_conversion",  action="store_true",  default=False,  help="Use or not the VSA converter",  )
    arg_parser.add_argument(  "--vsa_selection",  action="store_true",  default=False,  help="Use or not the VSA selector",  )
    arg_parser.add_argument(  "--context_superposition",  action="store_true",  default=False,  help="Use or not the VSA selector",  )
    arg_parser.add_argument(  "--program",  action="store_true",  default=False,  help="Program the model with golden weights",  )
    arg_parser.add_argument(  "--loss_fn", type=str, default="CosineLoss", help="Loss to use in the training"  )
    arg_parser.add_argument(  "--num_rules", type=int, default=5, help="Number of rules per each attribute"  )
    arg_parser.add_argument(  "--rule_selector_temperature",  type=float,  default=0.01,  help="Temperature used in the rule selector's softmax",  )
    arg_parser.add_argument(  "--rule_selector", type=str, default="weight", help="Can be sample or weight"  )
    arg_parser.add_argument(  "--shared_rules",  action="store_true",  default=False,  help="Share the same rules across different attributes",  )
    arg_parser.add_argument(  "--hidden_layers",  type=int,  default=3,  help="Number of hidden MLP layers to use in the neural model",  )
    arg_parser.add_argument(  "--nvsa-backend-d", type=int, default=1024, help="VSA dimension in backend"  )
    arg_parser.add_argument(  "--nvsa-backend-k", type=int, default=4, help="Number of blocks in VSA vectors"  )
    args = arg_parser.parse_args()
    return args
