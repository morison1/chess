from line import Line


def test_intersect_same_orientation():
    for vertical in [True, False]:
        line1 = Line(axis=0, small=-1, large=1, vertical=vertical)
        line2 = Line(axis=0, small=0, large=2, vertical=vertical)
        assert line1.intersect(line2)
        line3 = Line(axis=0, small=2, large=3, vertical=vertical)
        assert not line1.intersect(line3)
        line4 = Line(axis=5, small=2, large=3, vertical=vertical)
        assert not line1.intersect(line4)

def test_different_orientation():
    for vertical in [True, False]:
        line1 = Line(axis=0, small=-1, large=1, vertical=vertical)
        line2 = Line(axis=0, small=-1, large=1, vertical=not vertical)
        assert line1.intersect(line2)
        line3 = Line(axis=10, small=-1, large=1, vertical=not vertical)
        assert not line1.intersect(line3)

if __name__ == '__main__':
    line = Line(axis=1, small=0, large=10, vertical=True)
    for point in line:
        print(point)
    print(line.get_length())
    extended_line = line.extend_line(factor=2)
    print(extended_line.get_length())

    test_intersect_same_orientation()
    test_different_orientation()

