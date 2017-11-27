from Link import Link


class Node:

    def __init__(self, linkedNodes):
        self.links = []
        self.value = 0
        for node in linkedNodes:
            self.links.append(Link(1.0, node, self))

    def calculate(self):
        self.value = 0
        for link in self.links:
            self.value += link.start.value * link.weightFactor
