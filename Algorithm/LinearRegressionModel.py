# PA4 Q1
import random

import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import matplotlib.ticker as ticker

# Standardization  technique suggested by chat GTP
def standardization(feature):
    mean = np.mean(feature)
    std_dev = np.std(feature)
    standardized_feature = (feature - mean) / std_dev
    # The following line was use to get the mean and standard deviation for each feature type for use with the polynomial regression evaluator
    #print("first feature", feature[0], "mean", mean, "std_dev", std_dev)
    return standardized_feature

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

    #for fIndex in range(len(features)):
     #   print("feature ", fIndex, "mean", np.mean(features[fIndex]), "std_dev", np.std(features[fIndex]))

    #feature scaling done with help from chatgpt
    featureArray = np.array(features)
    scaledFeatureArray = np.array([standardization(feature) for feature in featureArray.T])
    scaledFeatures = scaledFeatureArray.T


    #print("numberOfBadValues", numberOfBadValues)
    return boards, scaledFeatures.tolist(), targets

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
        self.percentWinningPositionsIdentified = []
        self.percentWithinTwentyPoints = []
        # percentages that are in the correct bin, where the bin refers to the bins used by the random forest classifier
        self.percentWithinCorrectBin = []


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
        prediction = sum([self.get_features(x)[i] * self.get_weights()[i] for i in range(self.degree + 1)])

        return prediction

    def predict(self, x):
        return self.hypothesis(x)

    def loss(self, x, y):
        "*** YOUR CODE HERE ***"
        #print("lossweights", self.get_weights())
        #print("self.hypothesis(x)", self.hypothesis(x))
        #print("y", y)
        return (self.hypothesis(x) - y) ** 2

    def gradient(self, x, y):
        "*** YOUR CODE HERE ***"
        gradientValues = []
        for feature in self.features:
            gradientValues.append(((self.hypothesis(x) - y) * feature(x)))

        return gradientValues

    def train(self, regularization_strength=0.0):
        xBoards, xFeatures, ySamples = get_all_samples()

        testingXBoards = []
        testingxFeatures = []
        testingYSamples = []
        numberOfTestingDataPoints = 2000

        random.seed(42)
        for i in range(numberOfTestingDataPoints):
            index = random.randrange(0, len(xBoards), 1)
            testingXBoards.append(xBoards.pop(index))
            testingxFeatures.append(xFeatures.pop(index))
            testingYSamples.append(ySamples.pop(index))

        random.seed()

        '''#https://stackoverflow.com/questions/33626623/the-most-efficient-way-to-remove-first-n-elements-in-a-list
        testingXBoards = xBoards[:numberOfTestingDataPoints]
        xBoards = xBoards[numberOfTestingDataPoints:]
        testingxFeatures = xFeatures[:numberOfTestingDataPoints]
        xFeatures = xFeatures[numberOfTestingDataPoints:]
        testingYSamples = ySamples[:numberOfTestingDataPoints]
        ySamples = ySamples[numberOfTestingDataPoints:]'''

        numberOfLossSamples = 400
        epocs = 10
        iterationsPerEpoc = len(ySamples)
        numberOfIterations = epocs * iterationsPerEpoc

        # https://www.learndatasci.com/solutions/python-double-slash-operator-floor-division/#:~:text=In%20Python%2C%20we%20can%20perform,floor()%20function.
        lossStepSize = numberOfIterations // numberOfLossSamples

        iterations = 0
        for epoc in range(epocs):
            for i in range(iterationsPerEpoc):

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
                    #print("weights", self.get_weights())
                    #newWeights.append(self.weights[j] - self.learningRate * gradientValues[j])  # check here
                    regularization_term = regularization_strength * (self.get_weights()[j] ** 2)
                    regularized_gradient = gradientValues[j] + regularization_term
                    newWeights.append(
                        self.weights[j] - self.learningRate * regularized_gradient
                    )

                self.weights = newWeights

                # calculate and store loss
                if i % lossStepSize == 0:

                    totalLose = 0
                    winningPositionsIdentified = 0
                    withinTwentyPoints = 0
                    withinBin = 0
                    for sample in range(numberOfTestingDataPoints):
                        currentTestingFeatures = testingxFeatures[sample]
                        self.features[1] = lambda board: currentTestingFeatures[0]
                        self.features[2] = lambda board: currentTestingFeatures[1]
                        self.features[3] = lambda board: currentTestingFeatures[2]
                        self.features[4] = lambda board: currentTestingFeatures[3]
                        self.features[5] = lambda board: currentTestingFeatures[4]

                        totalLose += self.loss(testingXBoards[sample], testingYSamples[sample])

                        if self.hypothesis(sample) >= 0 and testingYSamples[sample] >= 0 or \
                            self.hypothesis(sample) < 0 and testingYSamples[sample] < 0:
                            winningPositionsIdentified += 1

                        if abs(self.hypothesis(sample) - testingYSamples[sample]) <= 25:
                            withinTwentyPoints += 1

                        if get_bin(self.hypothesis(sample)) == get_bin(testingYSamples[sample]):
                            withinBin += 1

                    self.eval_iters.append(iterations)
                    self.losses.append(totalLose / numberOfTestingDataPoints)
                    self.percentWinningPositionsIdentified.append(winningPositionsIdentified / numberOfTestingDataPoints)
                    self.percentWithinTwentyPoints.append(withinTwentyPoints / numberOfTestingDataPoints)
                    self.percentWithinCorrectBin.append(withinBin / numberOfTestingDataPoints)

                iterations += 1

        return self.weights


def get_bin(value):
    bins = [-float('inf'), -1000, -100, -23, 84, 500, 1000, float('inf')]
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i


def plot_loss_curve(eval_iters, losses, title = None):
    plt.plot(eval_iters, losses)

    #X axis scaling done with help from ChatGPT
    # Set the x-axis tick labels to display values in millions with one decimal point
    plt.xticks(ticks=plt.xticks()[0], labels=[f'{x / 1e6:.1f}M' for x in plt.xticks()[0]])
    plt.xlim(left=0)

    plt.xlabel("Iterations (Millions)")
    plt.ylabel("Loss")
    if title is not None:
        plt.title(title)
    plt.show()




linear_model = PolynomialRegressionModel(learning_rate=1 * 10 ** -6)
print("Weights", linear_model.train())


# part b
plot_loss_curve(linear_model.eval_iters, linear_model.losses, "Polynomial Regression Loss curve")

plt.plot(linear_model.eval_iters, linear_model.percentWinningPositionsIdentified)
plt.title("Polynomial Regression Percentage of Cases Our Model and Stockfish Agree on Who Is Winning")
plt.xlabel("Iterations (Millions)")
plt.ylabel("Percentage")
plt.xticks(ticks=plt.xticks()[0], labels=[f'{x / 1e6:.1f}M' for x in plt.xticks()[0]])
plt.yticks(ticks=plt.yticks()[0], labels=[f'{int(y * 100)}%' for y in plt.yticks()[0]])
plt.xlim(left=0)
plt.show()

plt.plot(linear_model.eval_iters, linear_model.percentWithinTwentyPoints)
plt.title("Polynomial Regression Percentage of Cases Our Model's Score is Within 25 Points of Stockfish")
plt.xlabel("Iterations (Millions)")
plt.ylabel("Percentage")
plt.xticks(ticks=plt.xticks()[0], labels=[f'{x / 1e6:.1f}M' for x in plt.xticks()[0]])
plt.yticks(ticks=plt.yticks()[0], labels=[f'{int(y * 100)}%' for y in plt.yticks()[0]])
plt.xlim(left=0)
plt.show()

plt.plot(linear_model.eval_iters, linear_model.percentWithinCorrectBin)
plt.title("Polynomial Regression Percentage of Cases Our Model's Predictions Are Within the Correct Bin")
plt.xlabel("Iterations (Millions)")
plt.ylabel("Percentage")
plt.xticks(ticks=plt.xticks()[0], labels=[f'{x / 1e6:.1f}M' for x in plt.xticks()[0]])
plt.yticks(ticks=plt.yticks()[0], labels=[f'{int(y * 100)}%' for y in plt.yticks()[0]])
plt.xlim(left=0)
plt.show()


