import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import _window_based_iterator


def test_window_based_iterator():
    f = lambda x: x
    for w1, w2, d in _window_based_iterator(list('123456789'), 3, f):
        if w1 == w2:
            assert d == 1
        else:
            expected = abs(int(w2) - int(w1))
            assert expected == d, f"w1: {w1}, w2: {w2}, d: {d}, expected: {expected}"


if __name__ == "__main__":
    test_window_based_iterator()
