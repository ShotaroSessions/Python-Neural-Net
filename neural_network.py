import numpy
import scipy.special


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

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):

        # Convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Calculate output error
        output_errors = targets - final_outputs

        # Calculate hidden layer error
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Update hidden to output weights
        self.who += self.lr * numpy.dot((output_errors * final_outputs *
                                        (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # Update input to hidden weights
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *
                                         (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):

        # Convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Calculate the signals emergin from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_outputs = numpy.dot(self.who, hidden_outputs)

        return final_outputs


if __name__ == '__main__':

    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print(n.query([1.0, 0.5, -1.5]))
