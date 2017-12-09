WIDTH = 12
HEIGHT = 12

# 12*12 grids
class Board:

    def __init__(self):
        self.top_wall = []
        self.bottom_wall = []
        self.left_wall = []
        for x in range(12):
            self.top_wall.append([0,x])
            self.bottom_wall.append([11,x])
            self.left_wall


