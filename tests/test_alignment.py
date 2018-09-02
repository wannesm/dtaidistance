import os
from pathlib import Path
import numpy as np
from dtaidistance import alignment


directory = None


def test_sequences1():
    s1 = "GATTACA"
    s2 = "GCATGCU"
    value, matrix = alignment.needleman_wunsch(s1, s2)
    matrix_sol = [
        [-0., -1., -2., -3., -4., -5., -6., -7.],
        [-1.,  1., -0., -1., -2., -3., -4., -5.],
        [-2., -0., -0.,  1., -0., -1., -2., -3.],
        [-3., -1., -1., -0.,  2.,  1., -0., -1.],
        [-4., -2., -2., -1.,  1.,  1., -0., -1.],
        [-5., -3., -3., -1., -0., -0., -0., -1.],
        [-6., -4., -2., -2., -1., -1.,  1., -0.],
        [-7., -5., -3., -1., -2., -2., -0., -0.]]
    assert value == 0.0
    assert np.array_equal(matrix, matrix_sol)


if __name__ == "__main__":
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    test_sequences1()
