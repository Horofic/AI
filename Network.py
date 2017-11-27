from Node import Node
from Link import Link
from math import *


class Network():

    def __init__(self, inputs, outputs):
        # create input nodes
        self.input = []
        for i in range(inputs):
            self.input.append(Node([]))

        # create output nodes
        self.output = []
        for i in range(outputs):
            self.output.append(Node(self.input))

        # create a list of all links
        self.links = []
        for node in self.output:
            for link in node.links:
                self.links.append(link)

    def inference(self, inputValues):
        # insert input node values
        count = 0
        for i in range(len(inputValues)):
            for val in inputValues[i]:
                self.input[count].value = val
                count += 1

        # calculate output node values
        answer = []
        for node in self.output:
            node.calculate()
            answer.append(node.value)

        # softmax activatie
        return self.softMax(answer)

    def softMax(self, output):
        result = []
        runningTotal = []
        for (i, x) in enumerate(output):
            fx = exp(-x)
            runningTotal.append(fx)
        for i in range(len(output)):
            fx = runningTotal[i] / sum(runningTotal)
            result.append(fx)
        return result

    def loss(self, y, ycorrect):
        # cross-entropy loss
        loss = 0
        for i in range(len(y)):
            loss -= ycorrect[i] * log(y[i]) + (1 - ycorrect[i]) * log(1 - y[i])
        return loss

    def computeAverageLoss(self, inputs, ycorrect):
        cumulativeLoss = 0
        for i in range(len(inputs)):
            cumulativeLoss += self.loss(self.inference(inputs[i]), ycorrect[i])
        return cumulativeLoss / len(inputs)

    def gradientDecentAverageLoss(self, link, inputs, ycorrect, delta):
        loss = self.computeAverageLoss(inputs, ycorrect)
        link.weightFactor += delta
        lossDeltaLoss = self.computeAverageLoss(
            inputs, ycorrect)

        link.gradient = (lossDeltaLoss - loss) / delta
        link.weightFactor -= delta

    def lineairRegression(self, inputs, ycorrect, learningRate, delta, wantedLoss):
        it = 0
        newLoss = self.computeAverageLoss(inputs, ycorrect)
        while (newLoss > wantedLoss):
            # calculate gradients
            for link in self.links:
                self.gradientDecentAverageLoss(
                    link, inputs, ycorrect, delta)

            # set new weightfactors
            for link in self.links:
                link.gradientDescent(learningRate)

            newLoss = self.computeAverageLoss(inputs, ycorrect)
            it += 1
            print("iteration " + str(it) + " loss: " + str(newLoss))
        # values found
        weightfactors = []
        for link in self.links:
            weightfactors.append(link.weightFactor)
        print("\nweightfactor(s): " + str(weightfactors) + "\n")

    def percentageGuessedCorrectly(self, inputs, ycorrect):
        totalCorrect = 0
        for i in range(len(inputs)):
            for x in range(len(inputs[i])):
                inputs[i][x] = round(inputs[i][x])
            if(inputs[i] == ycorrect[i]):
                totalCorrect += 1
        print("percentage correctly guessed : " +
              str((totalCorrect / len(inputs)) * 100) + "%")
