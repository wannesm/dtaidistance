#!/usr/bin/env python3
# encoding: utf-8
"""
hierarchical_clustering - Example

__author__ = "Wannes Meert"
__copyright__ = "Copyright 2017 KU Leuven, DTAI Research Group"
__license__ = "APL"

..
    Part of the DTAI distance code.

    Copyright 2016 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import sys
import os
import argparse
import logging
import math

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from dtaidistance import dtw

logger = logging.getLogger(__name__)


def size_cond(size):
    n = size
    r = 2
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))


def example_naivehierarchicalclustering():
    """Naive hierarchical clustering algorithm using DTW and based on .

    For a more efficient approach, check:
    Mueen, A and Keogh, E, Extracting Optimal Performance from Dynamic Time Warping,
    Tutorial, KDD 2016
    http://www.cs.unm.edu/~mueen/DTW.pdf

    :return: None
    """
    series = [
        np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0]),
        np.array([2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([2.0, 1.0, 1.0, 0.0, 0.0, 2.0, 3.0]),
        np.array([4.0, 2.0, 1.0, 0.0, 0.0, 1.0, 3.0])
    ]

    dists = dtw.distance_matrix_fast(series)
    print("Distance matrix:\n{}".format(dists))

    dists_cond = np.zeros(size_cond(len(series)))
    idx = 0
    for r in range(len(series)-1):
        dists_cond[idx:idx+len(series)-r-1] = dists[r, r+1:]
        idx += len(series)-r-1

    z = linkage(dists_cond, method='complete', metric='euclidean')
    print(z)

    fig, axes = plt.subplots(2, 1, figsize=(8, 3))
    for idx, serie in enumerate(series):
        serie += idx * 0.1
        axes[0].plot(serie, label=str(idx))
        axes[0].text(0 + 0.15 * (-1)**idx * idx, serie[0] + 0.15 * idx, idx)
        axes[0].add_line(Line2D([0, 0 + 0.15 * (-1)**idx * idx], [serie[0], serie[0] + 0.15 * idx],
                                linewidth=1, color='gray'))
    axes[0].legend(loc=1)
    dendrogram(z, ax=axes[1])
    plt.show(block=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Example hierarchical clustering')
    parser.add_argument('--verbose', '-v', action='count', help='Verbose output')
    args = parser.parse_args(argv)

    logger.setLevel(logging.ERROR - 10 * (0 if args.verbose is None else args.verbose))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    example_naivehierarchicalclustering()


if __name__ == "__main__":
    sys.exit(main())
