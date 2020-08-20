import numpy


class NeuralNetwork:
    """A Simple Neural Network with one hidden layer"""

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """Parameters are arrays of nodes and learning rate as a float"""

        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # Initialize random weights
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),
                                       (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),
                                       (self.onodes, self.hnodes))

    def train(self):
        pass

    def query(self):
        pass


if __name__ == '__main__':

    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

