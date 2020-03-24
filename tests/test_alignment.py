import os
from pathlib import Path
import numpy as np
from dtaidistance import alignment


directory = None


def test_sequences1():
    """Example from https://en.wikipedia.org/wiki/Needleman–Wunsch_algorithm . """
    s1 = "GATTACA"
    s2 = "GCATGCU"
    value, matrix = alignment.needleman_wunsch(s1, s2)
    algn, s1a1, s2a1 = alignment.best_alignment(matrix, s1, s2, gap='-')
    matrix_sol = [
        [-0., -1., -2., -3., -4., -5., -6., -7.],
        [-1.,  1., -0., -1., -2., -3., -4., -5.],
        [-2., -0., -0.,  1., -0., -1., -2., -3.],
        [-3., -1., -1., -0.,  2.,  1., -0., -1.],
        [-4., -2., -2., -1.,  1.,  1., -0., -1.],
        [-5., -3., -3., -1., -0., -0., -0., -1.],
        [-6., -4., -2., -2., -1., -1.,  1., -0.],
        [-7., -5., -3., -1., -2., -2., -0., -0.]]
    algn_sol1 = [['G', '-', 'A', 'T', 'T', 'A', 'C', 'A'], ['G', 'C', 'A', 'T', '-', 'G', 'C', 'U']]
    assert value == 0.0
    assert np.array_equal(matrix, matrix_sol)
    assert s1a1 == algn_sol1[0]
    assert s2a1 == algn_sol1[1]


def test_sequences2():
    s1 = "GAAAAAAAT"
    s2 = "GAAT"
    value, matrix = alignment.needleman_wunsch(s1, s2)
    algn, s1a1, s2a1 = alignment.best_alignment(matrix, s1, s2, gap='-')
    print(matrix)
    print(algn)
    print(s1a1)
    print(s2a1)
    algn_sol1 = [list('GAAAAAAAT'), list('GAA-----T')]
    assert s1a1 == algn_sol1[0]
    assert s2a1 == algn_sol1[1]


def test_sequences3():
    s1 = "GAAAAAAAT"
    s2 = "GAATA"
    value, matrix = alignment.needleman_wunsch(s1, s2)
    algn, s1a1, s2a1 = alignment.best_alignment(matrix, s1, s2, gap='-')
    print(matrix)
    print(algn)
    print(s1a1)
    print(s2a1)
    algn_sol1 = [list('GAAAAAAAT'), list('GAA-----T')]
    assert s1a1 == algn_sol1[0]
    assert s2a1 == algn_sol1[1]


def test_sequences_alignment():
    """Based on the following example.

    https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm#Advanced_presentation_of_algorithm

    """

    matrix = {
        ('A', 'A'): 10,
        ('A', 'G'): -1,
        ('A', 'C'): -3,
        ('A', 'T'): -4,
        ('G', 'G'): 7,
        ('G', 'C'): -5,
        ('G', 'T'): -3,
        ('C', 'C'): 9,
        ('C', 'T'): 0,
        ('T', 'T'): 8
    }
    substitution = alignment.substitution_from_dict(matrix)

    s1 = "AGACTAGTTAC"
    s2 = "CGAGACGT"

    value, matrix = alignment.needleman_wunsch(s2, s2, substitution=substitution)
    algn, s1a1, s2a1 = alignment.best_alignment(matrix, s1, s2, gap='-')

    print(matrix)
    print(algn)
    print(s1a1)
    print(s2a1)


if __name__ == "__main__":
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # test_sequences1()
    # test_sequences2()
    # test_sequences3()
    test_sequences_alignment()
