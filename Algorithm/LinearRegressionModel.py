# PA4 Q1
import random

import numpy as np
import matplotlib.pyplot as plt
import math
import csv


def get_all_samples():
    boards = []
    features = []
    targets = []
    numberOfBadValues = 0
    with open("/Users/dimitar/Documents/CS4100/Final Project/CS4100-AI-Project/Util/evaluation_results_dimitar.csv", 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header if present
        for row in reader:
            try:
                #ChatGPT helped create the code for for formatting the values read from the csv
                board = row[1]
                featureRow = [float(val) for val in row[2:7]]
                target = float(row[7])
            except(ValueError, IndexError):
                numberOfBadValues += 1

            boards.append(board)
            features.append(featureRow)
            targets.append(target)


    print("numberOfBadValues", numberOfBadValues)
    return boards, features, targets

class PolynomialRegressionModel():
    """
    Linear regression model with polynomial features (powers of x up to specified degree).
    x and y are real numbers. The goal is to fit y = hypothesis(x).
    """

    def __init__(self, degree=5, learning_rate=1e-3):
        "*** YOUR CODE HERE ***"
        self.degree = degree
        self.learningRate = learning_rate

        # https://stackoverflow.com/questions/452610/how-do-i-create-a-list-of-lambdas-in-a-list-comprehension-for-loop
        # I use lambda functions to return the x to the given power values
        self.features = [None] * 5

        self.features.insert(0, lambda board: 1)

        # creating list of random numbers method suggested by ChatGTP
        self.weights = [random.uniform(0, 0) for _ in range(degree + 1)]

        # These are used to store the numbers needed to graph the loss over the iterations.
        self.eval_iters = []  # stores the iteration index number
        self.losses = []  # stores the loss at the iteration index number with the same list index as in eval_iters

    def get_features(self, x):
        "*** YOUR CODE HERE ***"
        featureOutput = []
        for feature in self.features:
            featureOutput.append(feature(x))
        return featureOutput

    def get_weights(self):
        "*** YOUR CODE HERE ***"
        return self.weights

    def hypothesis(self, x):
        "*** YOUR CODE HERE ***"
        # https://www.geeksforgeeks.org/python-multiply-two-list/
        print(self.get_features(x))
        prediction = sum([self.get_features(x)[i] * self.get_weights()[i] for i in range(self.degree + 1)])

        return prediction

    def predict(self, x):
        return self.hypothesis(x)

    def loss(self, x, y):
        "*** YOUR CODE HERE ***"
        return (self.hypothesis(x) - y) ** 2

    def gradient(self, x, y):
        "*** YOUR CODE HERE ***"
        gradientValues = []
        for feature in self.features:
            gradientValues.append(((self.hypothesis(x) - y) * feature(x)))

        return gradientValues

    def train(self, evalset=None):
        xBoards, xFeatures, ySamples = get_all_samples()
        numberOfSamples = len(ySamples)

        numberOfIterations = 100000
        numberOfLossSamples = 100
        numberOfTestingDataPoints = 1000

        #https://stackoverflow.com/questions/33626623/the-most-efficient-way-to-remove-first-n-elements-in-a-list
        testingXBoards = xBoards[:numberOfTestingDataPoints]
        xBoards = xBoards[numberOfTestingDataPoints:]
        testingxFeatures = xFeatures[:numberOfTestingDataPoints]
        xFeatures = xFeatures[numberOfTestingDataPoints:]
        testingYSamples = ySamples[:numberOfTestingDataPoints]
        ySamples = ySamples[numberOfTestingDataPoints:]



        # https://www.learndatasci.com/solutions/python-double-slash-operator-floor-division/#:~:text=In%20Python%2C%20we%20can%20perform,floor()%20function.
        lossStepSize = numberOfIterations // numberOfLossSamples
        for i in range(numberOfIterations):

            currentX = xBoards[i]
            currentFeatures = xFeatures[i]
            currentY = ySamples[i]

            self.features[1] = lambda board: currentFeatures[0]
            self.features[2] = lambda board: currentFeatures[1]
            self.features[3] = lambda board: currentFeatures[2]
            self.features[4] = lambda board: currentFeatures[3]
            self.features[5] = lambda board: currentFeatures[4]

            newWeights = []

            gradientValues = self.gradient(currentX, currentY)

            for j in range(len(self.weights)):
                newWeights.append(self.weights[j] - self.learningRate * gradientValues[j])  # check here

            self.weights = newWeights

            # calculate and store loss
            if i % lossStepSize == 0:
                totalLose = 0
                for sample in range(numberOfTestingDataPoints):
                    currentTestingFeatures = testingxFeatures[sample]
                    self.features[1] = lambda board: currentTestingFeatures[0]
                    self.features[2] = lambda board: currentTestingFeatures[1]
                    self.features[3] = lambda board: currentTestingFeatures[2]
                    self.features[4] = lambda board: currentTestingFeatures[3]
                    self.features[5] = lambda board: currentTestingFeatures[4]

                    totalLose += self.loss(testingXBoards[sample], testingYSamples[sample])

                self.eval_iters.append(i)
                self.losses.append(totalLose / numberOfTestingDataPoints)


            elif i % 1000 == 0:
                print("i = ", 1)
                print("weights = ", self.get_weights())


        return self.weights



def plot_loss_curve(self, eval_iters, losses, title = None):
    plt.plot(eval_iters, losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    if title is not None:
        plt.title(title)
    plt.show()




linear_model = PolynomialRegressionModel(learning_rate=10 ** -4)
print("Q2a weights", linear_model.train())


# part b
plot_loss_curve(linear_model.eval_iters, linear_model.losses, "Q2b Loss curve")


'''# part c
    allSolutions = []
    sine_val = util.get_dataset("sine_val")

    learningRates = [10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5, 10 ** -4]

    for learningRate in learningRates:
        for degreeNum in range(1, 3, 1):
            individualSolution = []

            currentModel = PolynomialRegressionModel(degree=degreeNum, learning_rate=learningRate)

            sine_val.compute_average_loss(currentModel)

            individualSolution.append("Q2c Weights: ")
            individualSolution.append(currentModel.train(sine_train))
            individualSolution.append("Average Loss: ")
            individualSolution.append(sine_val.compute_average_loss(currentModel))
            individualSolution.append("degreeNum: ")
            individualSolution.append(degreeNum)
            individualSolution.append("learningRate: ")
            individualSolution.append(learningRate)

            allSolutions.append(individualSolution)

    # sorting technique suggested by chatgpt
    sortedSolutions = sorted(allSolutions, key=lambda x: x[3])
    for solution in sortedSolutions:
        print(solution)'''

