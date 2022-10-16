class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f'{self.__class__.__name__}(x={self.x},y={self.y})'


class Line:
    def __init__(self, axis: int, small: int, large: int, vertical: bool):
        self.axis = int(axis)
        self.small = int(small)
        self.large = int(large)
        self.vertical = vertical

    def extend_line(self, factor):
        delta = (factor - 1) * (self.large - self.small) / 2
        small = self.small - delta
        large = self.large + delta
        extended_line = Line(axis=self.axis, small=small, large=large, vertical=self.vertical)
        return extended_line

    def get_length(self):
        return self.large - self.small + 1

    def __len__(self):
        return self.large - self.small + 1

    def merge(self, other):
        """
        Assuming the lines intersect & on the same axis & they are both vertical (horizontal)
        """
        merged_lines = Line(axis=self.axis,
                            small=min(self.small, other.small),
                            large=max(self.large, other.large),
                            vertical=self.vertical)
        return merged_lines

    def intersect(self, other):
        if self.vertical == other.vertical:
            if self.axis != other.axis:
                return False
            first_line = self
            second_line = other
            if first_line.small > second_line.small:
                first_line, second_line = second_line, first_line
            # first_line.small  <= second_line.small
            return first_line.large >= second_line.small
        # one is vertical and one is horizontal
        first_check = other.small <= self.axis <= other.large
        second_check = self.small <= other.axis <= self.large
        return first_check and second_check

    def __iter__(self):
        if self.vertical:
            return (Point(x=self.axis, y=y) for y in range(self.small, self.large + 1))
        else:
            return (Point(x=x, y=self.axis) for x in range(self.small, self.large + 1))

    def __str__(self):
        return f'{self.__class__.__name__}(axis={self.axis},small={self.small},large={self.large},vertical={self.vertical}) length = {self.large - self.small}'

    def __repr__(self):
        return str(self)
