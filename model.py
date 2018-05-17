import torch

class Model:
    def __init__(self, x, y_labels):

        self.x = x # store the input data as an attribute for later modification

        h1_dim = 100 # number of nodes in hidden layer 1

        in_dim = int(x.get_shape()[1]) # input dimension - 784 (28 * 28) for MNIST
        out_dim = int(y_labels.get_shape()[1]) # output dimension - 10 classes (digits 0-9) for MNIST

        # Simple network with one hidden layer
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h1_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h1_dim, out_dim),
        )

        # A list of the trainable model parameters in the same order in which they appear in
        # the network when traversed input -> output (e.g. [weights b/w input and first hidden layer,
        # bias b/w input and hidden layer 1, ... , weights between last hidden layer and output, bias b/w
        # hidden layer and output])
        self.parameters = []

        for parameter in model.parameters():
            self.parameters.append(parameter)

        # Define the initial loss function as simple cross entropy, as the network will first be trained
        # on a single task (no Fisher Information incorporated into the loss function).
        self.loss = torch.nn.CrossEntropyLoss()

        # The output- predictions of classification likelihoods- is produced by running
        # the model on given input data.
        y_pred = self.model(x)

        # Check if predictions are correct- a prediction is correct if the index of the highest value in the prediction
        # output is the same as the index of the highest value in the label for that sample.
        # For example (MNIST):
        #   prediction: [0.1, 0.1, 0.05, 0.05, 0.2, 0.4, 0.1, 0.0, 0.0, 0.0]
        #   label:      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        #
        #   This would be a correct prediction- the sixth entry (index 5) in each array holds the highest
        #   value
        #
        # Using dimension 1 as an argument allows us to get the index of the highest valued column in each row,
        # which practically translates to getting the maximum predicted value for each sample and comparing that
        # value to the label for that sample. NOTE: torch.eq() returns 0 if args not equal, 1 if args equal
        correct_prediction = torch.eq(torch.argmax(y_pred, 1), torch.argmax(y_labels, 1))

        # The overall accuracy of the model's predictions is the sum of its accurate predictions (valued at 1)
        # divided by the number of predictions it made (incorrect predictions evaluate to 0)
        self.accuracy = torch.mean(correct_prediction)


