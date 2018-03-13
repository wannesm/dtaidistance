import os
import sys
import math
import tempfile
import pytest
import logging
from pathlib import Path
import numpy as np
from dtaidistance import dtw, clustering


logger = logging.getLogger("be.kuleuven.dtai.distance")


def test_bug1(directory=None):
    series = np.matrix([
        [0., 0, 1, 2, 1, 0, 1, 0, 0],
        [0., 1, 2, 0, 0, 0, 0, 0, 0],
        [1., 2, 0, 0, 0, 0, 0, 1, 1],
        [0., 0, 1, 2, 1, 0, 1, 0, 0],
        [0., 1, 2, 0, 0, 0, 0, 0, 0],
        [1., 2, 0, 0, 0, 0, 0, 1, 1]])
    model = clustering.LinkageTree(dtw.distance_matrix_fast, {})
    cluster_idx = model.fit(series)

    if directory:
        hierarchy_fn = directory / "hierarchy.png"
    else:
        file = tempfile.NamedTemporaryFile()
        hierarchy_fn = Path(file.name + "_hierarchy.png")
    model.plot(hierarchy_fn)
    print("Figure saved to", hierarchy_fn)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.propagate = 0
    test_bug1(directory=Path.home() / "Desktop/")
