from Network import Network


def oneHotEncode(input):
    newCorrect = []
    for x in input:
        if(x == "O"):
            newCorrect.append([1, 0])
        elif(x == "X"):
            newCorrect.append([0, 1])
    return newCorrect


if __name__ == "__main__":
    # make a network with 9 input- and 2 output nodes
    ntwrk = Network(9, 2)

    # nice array of values to train my network with
    trainingSetX = [
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ],
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ],
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]
    ]

    # nice array of values to check if my network actually learned something
    testSetX = [
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ],
        [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 0]
        ],
        [
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 1]
        ],
        [
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0]
        ]
    ]

    # one hot encode O to [1, 0] and X to [0, 1]
    correctSet = ["O", "O", "X", "X"]
    ycorrect = oneHotEncode(correctSet)

    # do the training
    ntwrk.lineairRegression(trainingSetX, ycorrect, 1, (1 / 10000000), 0.01)

    # consult the trained network with the test values
    inferencedValues = []
    for i in range(len(testSetX)):
        inferenced = ntwrk.inference(testSetX[i])
        inferencedValues.append(inferenced)
        print("network answer : " + str(inferenced) +
              " correct answer : " + str(ycorrect[i]))

    print("\ntest values")
    ntwrk.percentageGuessedCorrectly(inferencedValues, ycorrect)

    inferencedValues = []
    for x in range(len(trainingSetX)):
        inferenced = ntwrk.inference(trainingSetX[x])
        inferencedValues.append(inferenced)

    print("\ntraining values")
    ntwrk.percentageGuessedCorrectly(inferencedValues, ycorrect)
