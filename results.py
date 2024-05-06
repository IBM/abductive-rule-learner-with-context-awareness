# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

import numpy as np
from collections import defaultdict
import os
from arlc.utils.parsing import eval_parse_args
import json


def main():
    print("\n")
    args = eval_parse_args()
    res = defaultdict(list)
    for i in range(1, args.seeds + 1):
        with open(os.path.join(args.path, f"{i}/ckpt/eval.json")) as f:
            dat = json.load(f)
        for k, v in dat.items():
            res[k].append(v)
    for k, v in res.items():
        print(f"{k}\t\t{np.mean(v)} ({np.std(v)})")

    mean = np.mean(sum(res.values(), []))
    std = np.mean([np.std(x) for x in res.values()])
    print("\nLaTex table entry:")
    print(
        " & ".join([f"${np.mean(v):.1f}^{{\pm{np.std(v):.1f}}}$" for v in res.values()])
        + f" & ${mean:.1f}^{{\pm{std:.1f})}}$"
    )
    print("\n")


if __name__ == "__main__":
    main()
