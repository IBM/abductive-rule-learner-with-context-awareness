#
# Copyright (c) IBM Corp. 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
