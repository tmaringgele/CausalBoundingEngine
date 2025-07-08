class Data:
    def __init__(self, X, Y, Z=None):
        self.X = X
        self.Y = Y
        self.Z = Z

    def unpack(self):
        if self.Z is not None:
            return {'X': self.X, 'Y': self.Y, 'Z': self.Z}
        else:
            return {'X': self.X, 'Y': self.Y}
