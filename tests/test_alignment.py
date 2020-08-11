import os
from pathlib import Path

import pytest

from dtaidistance import alignment, util_numpy
from dtaidistance.util import read_substitution_matrix


numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")
directory = Path()


@numpyonly
def test_sequences1():
    """Example from https://en.wikipedia.org/wiki/Needleman–Wunsch_algorithm . """
    with util_numpy.test_uses_numpy() as np:
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
    algn_sol1 = [list('GAAAAAAAT'), list('GAA-----T')]
    assert s1a1 == algn_sol1[0]
    assert s2a1 == algn_sol1[1]


def test_sequences3():
    s1 = "GAAAAAAAT"
    s2 = "GAATA"
    value, matrix = alignment.needleman_wunsch(s1, s2)
    algn, s1a1, s2a1 = alignment.best_alignment(matrix, s1, s2, gap='-')
    algn_sol1 = [list('GAA-AAAAAT'), list('GAATA-----')]
    assert s1a1 == algn_sol1[0]
    assert s2a1 == algn_sol1[1]


def test_sequences4():
    s1 = "AGACTAGTTACC"
    s2 = "CGAGACGTC"
    value, matrix = alignment.needleman_wunsch(s1, s2)
    algn, s1a1, s2a1 = alignment.best_alignment(matrix, s1, s2, gap='-')
    # print(matrix)
    # print(algn)
    # print(s1a1)
    # print(s2a1)
    algn_sol1 = [list("--AGACTAGTTACC"), list("CGAGAC--GT--C-")]
    assert s1a1 == algn_sol1[0]
    assert s2a1 == algn_sol1[1]


def test_sequences_custom():

    matrix = read_substitution_matrix(
        Path(__file__).parent / "rsrc" / "substitution.txt"
    )
    substitution = alignment.make_substitution_fn(matrix)

    s1 = "CCAGG"
    s2 = "CCGA"

    value, matrix = alignment.needleman_wunsch(s1, s2)
    algn, s1a1, s2a1 = alignment.best_alignment(matrix, s1, s2, gap='-')
    algn_sol1 = [list("CCAGG"), list("CC-GA")]
    assert s1a1 == algn_sol1[0]
    assert s2a1 == algn_sol1[1]

    value, matrix = alignment.needleman_wunsch(s1, s2, substitution=substitution)
    algn, s1a2, s2a2 = alignment.best_alignment(matrix, s1, s2, gap='-')
    algn_sol2 = [list("CC-AGG"), list("CCGA--")]
    assert s1a2 == algn_sol2[0]
    assert s2a2 == algn_sol2[1]


def test_sequences_blosum():
    matrix = read_substitution_matrix(
        Path(__file__).parent / "rsrc" / "substitution.txt"
    )
    substitution = alignment.make_substitution_fn(matrix)
    s1 = "AGACTAGTTAC"
    s2 = "CGAGACGT"
    value, matrix = alignment.needleman_wunsch(s1, s2, substitution=substitution)
    algn, s1a1, s2a1 = alignment.best_alignment(matrix, s1, s2, gap='-')
    algn_sol1 = [list('--AGACTAGTTAC'),
                 list('CGAGAC--GT---')]
    assert s1a1 == algn_sol1[0]
    assert s2a1 == algn_sol1[1]               


def test_substitution_function():
    matrix = read_substitution_matrix(
        Path(__file__).parent / "rsrc" / "substitution.txt"
    )
    substitution = alignment.make_substitution_fn(matrix, gap=0.5)
    assert substitution('A', 'B') == (1, 0.5)
    assert substitution('A', 'A') == (-20.0, 0.5)
    assert substitution('A', 'C') == (3.0, 0.5)

    # test just changing gap function
    substitution2 = alignment.make_substitution_fn({}, gap=0.5)
    assert substitution2('A', 'A') == (-1, 0.5)
    assert substitution2('G', 'C') == (1, 0.5)
    assert alignment._default_substitution_fn('A', 'A') == (-1, 1)
    assert alignment._default_substitution_fn('G', 'C') == (1, 1)


if __name__ == "__main__":
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    test_sequences1()
    test_sequences2()
    test_sequences3()
    test_sequences4()
    test_sequences_custom()
    test_sequences_blosum()
    test_substitution_function()
