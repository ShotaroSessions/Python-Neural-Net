import numpy
import scipy.special
import matplotlib.pyplot


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
        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_outputs = numpy.dot(self.who, hidden_outputs)

        return final_outputs


if __name__ == '__main__':
    """Neural net demo that loads the mnist dataset"""

    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Loads the mnist training dataset
    training_data_file = open("mnist_train_100.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # Prepare the data set
    for digit in training_data_list:
        all_values = digit.split(',')
        # Scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # Create the target output values
        targets = numpy.zeros(output_nodes) + 0.01
        # Target label at all_values[0]
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

    # Test the neural network
    test_data_file = open("mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    all_values = test_data_list[1].split(',')
    print(all_values[0])
    image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()
