# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

import json
import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict as dd

import arlc.utils.raven.env as reasoning_env
from arlc.utils.averagemeter import AverageMeter
from arlc.utils.checkpath import check_paths, save_checkpoint
from arlc.datasets import GeneralIRAVENDataset
from arlc.execution import RuleLevelReasoner
from arlc.selection import RuleSelector
from arlc.utils.vsa import generate_nvsa_codebooks
import arlc.losses as losses
from arlc.utils.raven.raven_one_hot import create_one_hot
from arlc.utils.parsing import parse_args
from arlc.utils.general import iravenx_rule_map, iravenx_index_map


def compute_loss_and_scores(
    outputs,
    tests,
    candidates,
    targets,
    loss_fn,
    distribute,
    params=None,
    confounders=False,
    attr_entropy=None,
    use_entropy=False,
):
    loss = 0
    scores = 0
    att_scores = dd(lambda: 0)
    for attr in outputs._fields:
        # in constellations without position, do not compute a loss on it
        if not distribute and (attr == "position" or attr == "number"):
            continue
        # if confounders is turned off, disregard confounding attributes
        if (attr == "angle" or "confounder" in attr) and not confounders:
            pass
        # add to the loss the contribution of attr
        else:
            # attribute entropy regularization
            entropy_reg = np.clip(attr_entropy[attr], 0.1, 10) if use_entropy else 1
            loss += (
                loss_fn(
                    getattr(outputs, attr),
                    torch.cat(
                        (
                            getattr(tests, attr),
                            getattr(candidates, attr)[
                                torch.arange(getattr(candidates, attr).shape[0]),
                                targets,
                            ].unsqueeze(1),
                        ),
                        dim=1,
                    ),
                ).mean(dim=-1)
                / entropy_reg
            )
        # compute attribute scores
        att_scores[attr] = loss_fn.score(
            getattr(outputs, attr)[:, -1].unsqueeze(1).repeat(1, 8, 1),
            getattr(candidates, attr),
        )
        scores += att_scores[attr] / entropy_reg

    return loss, scores, [att_scores[att] for att in ["type", "color", "size"]]


def train(args, env, device, confounders=False):
    """
    Training and validation of learnable NVSA backend
    """

    def inference_epoch(epoch, loader, train=True):
        if train:
            model.train()
            if args.config == "in_out_four":
                model2.train()
            rule_selector.train()
        else:
            model.eval()
            if args.config == "in_out_four":
                model2.eval()
            rule_selector.eval()

        # Define tracking meters
        loss_avg = AverageMeter("Loss", ":.3f")
        acc_avg = AverageMeter("Accuracy", ":.3f")

        for counter, (extracted, targets, all_action_rule) in enumerate(tqdm(loader)):
            extracted, targets, all_action_rule = (
                extracted.to(device),
                targets.to(device),
                all_action_rule.to(device),
            )
            att_logprob = create_one_hot(extracted, args.config, args.sigma)
            model_output = {k: v.to(device) for k, v in att_logprob.items()}
            scene_prob, _ = env.prepare(model_output)
            if args.config in ["center_single", "distribute_four", "distribute_nine"]:
                outputs, candidates, tests = model(
                    scene_prob, targets, distribute=distribute
                )
                outputs, attr_entropy = rule_selector(
                    outputs, tests, candidates, targets
                )
                loss, scores, _ = compute_loss_and_scores(
                    outputs,
                    tests,
                    candidates,
                    targets,
                    loss_fn,
                    "distribute" in args.config,
                    [p.data for p in model.parameters() if p.requires_grad],
                    confounders=confounders,
                    attr_entropy=attr_entropy,
                    use_entropy=args.entropy,
                )
            else:
                outputs1, candidates1, tests1 = model(
                    scene_prob[0], distribute=distribute
                )
                outputs1, attr_entropy1 = rule_selector(
                    outputs1, tests1, candidates1, targets
                )
                if args.config == "in_out_four":
                    outputs2, candidates2, tests2 = model2(scene_prob[1])
                else:
                    outputs2, candidates2, tests2 = model(scene_prob[1])

                outputs2, attr_entropy2 = rule_selector(
                    outputs2, tests2, candidates2, targets
                )
                loss1, scores1, _ = compute_loss_and_scores(
                    outputs1,
                    tests1,
                    candidates1,
                    targets,
                    loss_fn,
                    distribute=args.config == "in_out_four",
                    confounders=confounders,
                    attr_entropy=attr_entropy1,
                    use_entropy=args.entropy,
                )
                loss2, scores2, _ = compute_loss_and_scores(
                    outputs2,
                    tests2,
                    candidates2,
                    targets,
                    loss_fn,
                    distribute=False,
                    confounders=confounders,
                    attr_entropy=attr_entropy2,
                    use_entropy=args.entropy,
                )
                loss = 0.8 * loss1 + 0.2 * loss2
                scores = 0.8 * scores1 + 0.2 * scores2

            predictions = torch.argmax(scores, dim=-1)
            accuracy = ((predictions == targets).sum() / len(targets)) * 100
            loss_avg.update(loss.item(), extracted.size(0))
            acc_avg.update(accuracy.item(), extracted.size(0))
            acc_avg.update(accuracy.item(), extracted.size(0))

            if train:
                optimizer.zero_grad()
                loss.backward()
                if args.clip:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=train_param, max_norm=args.clip, norm_type=2.0
                    )
                optimizer.step()

        if train:
            print(
                "Epoch {}, Total Iter: {}, Train Avg Loss: {:.6f}, Train Avg Accuracy: {:.6f}".format(
                    epoch, counter, loss_avg.avg, acc_avg.avg
                )
            )
            writer.add_scalar("loss/training", loss_avg.avg, epoch)
            writer.add_scalar("accuracy/training", acc_avg.avg, epoch)
        else:
            print(
                "Epoch {}, Valid Avg Loss: {:.6f}, Valid Avg Acc: {:.4f}".format(
                    epoch, loss_avg.avg, acc_avg.avg
                )
            )
            writer.add_scalar("loss/validation", loss_avg.avg, epoch)
            writer.add_scalar("accuracy/validation", acc_avg.avg, epoch)
        for r in model.rules_set.rules:
            print(str(r.rule))
        return acc_avg.avg

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    writer = SummaryWriter(args.log_dir)

    # Init model
    model = RuleLevelReasoner(
        args.device,
        args.config,
        model=args.model,
        hidden_layers=args.hidden_layers,
        dictionary=args.backend_cb,
        vsa_conversion=args.vsa_conversion,
        vsa_selection=args.vsa_selection,
        context_superposition=args.context_superposition,
        num_rules=args.num_rules,
        shared_rules=args.shared_rules,
        program=args.program,
        num_terms=args.num_terms,
        n=args.n,
    )
    model.to(args.device)
    if args.config == "in_out_four":
        model2 = RuleLevelReasoner(
            args.device,
            "center_single",
            model=args.model,
            hidden_layers=args.hidden_layers,
            dictionary=args.backend_cb,
            vsa_conversion=args.vsa_conversion,
            vsa_selection=args.vsa_selection,
            context_superposition=args.context_superposition,
            num_rules=args.num_rules,
            shared_rules=args.shared_rules,
            program=args.program,
            num_terms=args.num_terms,
            n=args.n,
        )
        model2.to(args.device)

    distribute = "distribute" in args.config or "in_out_four" == args.config
    # Init loss
    loss_fn = getattr(losses, args.loss_fn)()

    rule_selector = RuleSelector(
        loss_fn, args.rule_selector_temperature, rule_selector=args.rule_selector
    )

    # Init optimizers
    train_param = list(model.parameters())
    if args.config == "in_out_four":
        train_param += list(model2.parameters())
    optimizer = optim.AdamW(train_param, args.lr, weight_decay=args.weight_decay)

    # Load all checkpoints
    rule_path = os.path.join(args.resume, "checkpoint.pth.tar")
    if os.path.isfile(rule_path):
        checkpoint = torch.load(rule_path)
        model.load_state_dict(checkpoint["state_dict_model"])
        if args.config == "in_out_four":
            model2.load_state_dict(checkpoint["state_dict_model2"])
        best_accuracy = checkpoint["best_accuracy"]
        start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' at Epoch {:.3f}".format(
                rule_path, checkpoint["epoch"]
            )
        )
    else:
        best_accuracy = 0
        start_epoch = 0

    # Dataset loader
    train_set = GeneralIRAVENDataset(
        "train",
        args.data_dir,
        constellation_filter=args.config,
        rule_filter=args.gen_rule,
        attribute_filter=args.gen_attribute,
        n_train=args.n_train,
        maxval=args.dyn_range,
        partition=args.partition,
        n=args.n,
        n_confounders=args.orientation_confounder,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_set = GeneralIRAVENDataset(
        "val",
        args.data_dir,
        constellation_filter=args.config,
        rule_filter=args.gen_rule,
        attribute_filter=args.gen_attribute,
        n_train=args.n_train,
        maxval=args.dyn_range,
        partition=args.partition,
        n=args.n,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 15, num_workers=args.num_workers
    )

    # training loop starts
    for epoch in range(start_epoch, args.epochs):
        inference_epoch(epoch, loader=train_loader, train=True)
        with torch.no_grad():
            accuracy = inference_epoch(epoch, loader=val_loader, train=False)

        # store model(s)
        is_best = accuracy > best_accuracy
        best_accuracy = max(accuracy, best_accuracy)
        if args.config == "in_out_four":
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict_model": model.state_dict(),
                    "state_dict_model2": model2.state_dict(),
                    "best_accuracy": accuracy,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                savedir=args.checkpoint_dir,
            )
        else:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict_model": model.state_dict(),
                    "best_accuracy": best_accuracy,
                    "accuracy": accuracy,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                savedir=args.checkpoint_dir,
            )
    return writer


def test(args, env, device, writer=None, dset="RAVEN", confounders=False):
    """
    Testing of NVSA backend
    """

    def test_epoch():
        model.eval()
        if args.config == "in_out_four":
            model2.eval()
        rule_selector.eval()

        loss_avg = AverageMeter("Loss", ":.3f")
        acc_avg = AverageMeter("Accuracy", ":.3f")
        rule_acc_avg = {
            rule: AverageMeter("Accuracy", ":.3f") for rule in iravenx_rule_map.keys()
        }

        for extracted, targets, all_action_rule in tqdm(test_loader):
            extracted, targets, all_action_rule = (
                extracted.to(device),
                targets.to(device),
                all_action_rule.to(device),
            )
            att_logprob = create_one_hot(extracted, args.config, args.sigma)
            model_output = {k: v.to(device) for k, v in att_logprob.items()}
            scene_prob, _ = env.prepare(model_output)
            if args.config in ["center_single", "distribute_four", "distribute_nine"]:
                outputs, candidates, tests = model(scene_prob, distribute=distribute)
                outputs, attr_entropy = rule_selector(outputs, tests)
                loss, scores, attscores = compute_loss_and_scores(
                    outputs,
                    tests,
                    candidates,
                    targets,
                    loss_fn,
                    "distribute" in args.config,
                    [p.data for p in model.parameters() if p.requires_grad],
                    confounders=confounders,
                    attr_entropy=attr_entropy,
                    use_entropy=args.entropy,
                )

            else:
                outputs1, candidates1, tests1 = model(
                    scene_prob[0], distribute=distribute
                )
                outputs1, attr_entropy1 = rule_selector(outputs1, tests1)
                if args.config == "in_out_four":
                    outputs2, candidates2, tests2 = model2(
                        scene_prob[1], distribute=False
                    )
                else:
                    outputs2, candidates2, tests2 = model(
                        scene_prob[1], distribute=False
                    )
                outputs2, attr_entropy2 = rule_selector(outputs2, tests2)
                loss1, scores1, _ = compute_loss_and_scores(
                    outputs1,
                    tests1,
                    candidates1,
                    targets,
                    loss_fn,
                    distribute=args.config == "in_out_four",
                    confounders=confounders,
                    attr_entropy=attr_entropy1,
                    use_entropy=args.entropy,
                )
                loss2, scores2, _ = compute_loss_and_scores(
                    outputs2,
                    tests2,
                    candidates2,
                    targets,
                    loss_fn,
                    distribute=False,
                    confounders=confounders,
                    attr_entropy=attr_entropy2,
                    use_entropy=args.entropy,
                )
                loss = (loss1 + loss2) / 2
                scores = (scores1 + scores2) / 2

            # accuracy and loss computation
            predictions = torch.argmax(scores, dim=-1)
            accuracy = ((predictions == targets).sum() / len(targets)) * 100
            loss_avg.update(loss.item(), extracted.size(0))
            acc_avg.update(accuracy.item(), extracted.size(0))
            ##### <- rule accuracy computation
            if args.evaluate_rule:
                expanded_rules = all_action_rule[:, 1:]
                batch_size = all_action_rule.shape[0]
                for rule in iravenx_index_map.keys():
                    total = 0
                    correct = 0
                    rule_mask = expanded_rules == rule
                    if rule_mask.any():
                        if args.sigma >= 0:
                            for attribute_idx in range(0, 3):
                                max_scores = attscores[attribute_idx].max(dim=-1).values
                                not_max_entropy = ~torch.all(
                                    attscores[attribute_idx]
                                    == attscores[attribute_idx][:, 0:1],
                                    dim=1,
                                )
                                correct += (
                                    (
                                        torch.logical_and(
                                            attscores[attribute_idx][
                                                torch.arange(batch_size), targets
                                            ]
                                            == max_scores,
                                            torch.logical_and(
                                                rule_mask[:, attribute_idx],
                                                not_max_entropy,
                                            ),
                                        )
                                    )
                                    .sum()
                                    .item()
                                )
                                total += rule_mask[:, attribute_idx].sum().item()
                        else:
                            correct += (
                                (rule_mask.sum(dim=-1) * (predictions == targets))
                                .sum()
                                .item()
                            )
                            total += rule_mask.sum().item()

                    rule_acc = correct / total if total else 0
                    rule_acc_avg[iravenx_index_map[rule]].update(
                        rule_acc, extracted.size(0)
                    )
            ##### -> rule accuracy computation

        # Save final result as npz (and potentially in Tensorboard)
        if args.resume == "":
            if writer is not None:
                writer.add_scalar("accuracy/testing-{}".format(dset), acc_avg.avg, 0)
                np.savez(
                    args.save_dir + "result_{:}.npz".format(dset), loss=acc_avg.avg
                )
            else:
                args.save_dir = args.resume.replace("ckpt/", "save/")
                np.savez(
                    args.save_dir + "result_{:}.npz".format(dset), loss=acc_avg.avg
                )

        print("Test Avg Accuracy: {:.4f}".format(acc_avg.avg))
        if args.evaluate_rule:
            for rule in iravenx_rule_map.keys():
                print(f"Rule {rule} Avg Accuracy: {rule_acc_avg[rule].avg * 100:.2f}")
        for r in model.rules_set.rules:
            print(str(r.rule))
        return {
            **{"acc": acc_avg.avg},
            **{rule: rule_acc_avg[rule].avg * 100 for rule in iravenx_rule_map.keys()},
        }

    # Load all checkpoint
    model_path = os.path.join(args.resume, "model_best.pth.tar")
    # model_path = os.path.join(args.resume, "checkpoint.pth.tar")
    print(model_path)
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        print(
            "=> loaded checkpoint '{}', epoch {}, with accuracy {:.3f}".format(
                model_path, checkpoint["epoch"], checkpoint["best_accuracy"]
            )
        )
    else:
        print(
            f"Careful! The model is not loaded from checkpoint. Program is: {args.program}"
        )
        # raise ValueError("No checkpoint found at {:}".format(model_path))
    test_acc = dict()
    configs = [
        "center_single",
        # "distribute_four",
        # "distribute_nine",
        # "left_right",
        # "up_down",
        # "in_out_single",
        # "in_out_four",
    ]
    for config in configs:
        args.config = config
        env = reasoning_env.get_env(args.configs_map[args.config], device)
        # Init the model
        model = RuleLevelReasoner(
            args.device,
            config,
            model=args.model,
            hidden_layers=args.hidden_layers,
            dictionary=args.backend_cb,
            vsa_conversion=args.vsa_conversion,
            vsa_selection=args.vsa_selection,
            context_superposition=args.context_superposition,
            num_rules=args.num_rules,
            shared_rules=args.shared_rules,
            program=args.program,
            num_terms=args.num_terms,
            n=args.n,
        )
        model.to(device)
        if not args.program:
            model.load_state_dict(checkpoint["state_dict_model"])
        if args.data_dir == "/dccstor/saentis/data/I-RAVEN":
            model.anneal_softmax()
        if config == "in_out_four":
            model2 = RuleLevelReasoner(
                args.device,
                "center_single",
                model=args.model,
                hidden_layers=args.hidden_layers,
                dictionary=args.backend_cb,
                vsa_conversion=args.vsa_conversion,
                vsa_selection=args.vsa_selection,
                context_superposition=args.context_superposition,
                num_rules=args.num_rules,
                shared_rules=args.shared_rules,
                program=args.program,
                num_terms=args.num_terms,
                n=args.n,
            )
            model2.to(device)
            if not args.program:
                model2.load_state_dict(checkpoint["state_dict_model"])
        distribute = "distribute" in config or "in_out_four" == config
        # Init loss
        loss_fn = getattr(losses, args.loss_fn)()

        rule_selector = RuleSelector(
            loss_fn, args.rule_selector_temperature, rule_selector=args.rule_selector
        )

        # Dataset loader
        test_set = GeneralIRAVENDataset(
            "test",
            args.data_dir,
            constellation_filter=config,
            rule_filter=args.gen_rule,
            attribute_filter=args.gen_attribute,
            maxval=args.dyn_range,
            partition=args.partition,
            n=args.n,
            n_confounders=args.orientation_confounder,
        )
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, num_workers=args.num_workers
        )
        print("Evaluating on {}".format(config))
        with torch.no_grad():
            acc = test_epoch()
        test_acc[config] = acc

    with open(os.path.join(args.resume, f"eval.json"), "w") as fp:
        json.dump(test_acc, fp)
    return writer


def main():
    args = parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cuda = torch.cuda.is_available()

    # Use a rng for reproducible results
    rng = np.random.default_rng(seed=args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Load or define new codebooks
    backend_cb_cont, backend_cb_discrete = generate_nvsa_codebooks(args, rng)

    args.backend_cb_discrete = backend_cb_discrete
    args.backend_cb_cont = backend_cb_cont

    print(f"Sigma: {args.sigma}")

    if args.model == "LearnableFormula":
        args.backend_cb = backend_cb_cont
    else:
        args.backend_cb = backend_cb_discrete

    # backend for training/testing
    input_configs = [
        "center_single",
        "left_right",
        "up_down",
        "in_out_single",
        "distribute_four",
        "in_out_four",
        "distribute_nine",
    ]
    output_configs = [
        "center_single",
        "left_center_single_right_center_single",
        "up_center_single_down_center_single",
        "in_center_single_out_center_single",
        "distribute_four",
        "in_distribute_four_out_center_single",
        "distribute_nine",
    ]
    args.configs_map = dict(zip(input_configs, output_configs))

    env = reasoning_env.get_env(args.configs_map[args.config], args.device)

    if args.mode == "train":
        args.exp_dir = os.path.join(args.exp_dir, args.run_name, str(args.seed))
        args.checkpoint_dir = os.path.join(args.exp_dir, "ckpt")
        args.save_dir = os.path.join(args.exp_dir, "save")
        args.log_dir = os.path.join(args.exp_dir, "log")
        check_paths(args)

        # Run the actual training
        writer = train(args, env, args.device, confounders=args.orientation_confounder)

        # Do final testing
        args.resume = args.checkpoint_dir
        writer = test(
            args, env, args.device, writer, confounders=args.orientation_confounder
        )

        writer.close()

    elif args.mode == "test":
        test(args, env, args.device, confounders=args.orientation_confounder)


if __name__ == "__main__":
    main()
