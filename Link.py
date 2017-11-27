class Link:

    def __init__(self, weightFactor, start, end):
        self.weightFactor = weightFactor
        self.start = start
        self.end = end
        self.gradient = 1

    def gradientDescent(self, learningRate):
        self.weightFactor = self.weightFactor - learningRate * self.gradient
