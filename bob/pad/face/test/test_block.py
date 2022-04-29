import numpy

A_org = numpy.array(range(1, 17), "float64").reshape((4, 4))
A_ans_0_3D = numpy.array(
    [
        [[1, 2], [5, 6]],
        [[3, 4], [7, 8]],
        [[9, 10], [13, 14]],
        [[11, 12], [15, 16]],
    ],
    "float64",
)
A_ans_0_4D = numpy.array(
    [
        [[[1, 2], [5, 6]], [[3, 4], [7, 8]]],
        [[[9, 10], [13, 14]], [[11, 12], [15, 16]]],
    ],
    "float64",
)

from bob.pad.face.utils.load_utils import block


def test_block():
    B = block(A_org, (2, 2), (0, 0))

    assert (B == A_ans_0_4D).all()

    B = block(A_org, (2, 2), (0, 0), flat=True)
    assert (B == A_ans_0_3D).all()
