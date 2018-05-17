import torch

class Model:
    def __init__(self, x, y_):

        h1_dim = 100 # number of nodes in hidden layer 1

        in_dim = int(x.get_shape()[1]) # input dimension - 784 (28 * 28) for MNIST
        out_dim = int(y_.get_shape()[1]) # output dimension - 10 classes (digits 0-9) for MNIST

        # Simple network with one hidden layer
        model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h1_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h1_dim, out_dim),
        )

        # a list of the trainable model parameters in the same order in which they appear in
        # the network when traversed input -> output (e.g. [weights b/w input and first hidden layer,
        # bias b/w input and hidden layer 1, ... , weights between last hidden layer and output, bias b/w
        # hidden layer and output])
        self.parameters = model.parameters()

        


class Model:
    def __init__(self, x, y_):

        in_dim = int(x.get_shape()[1]) # 784 for MNIST
        out_dim = int(y_.get_shape()[1]) # 10 for MNIST

        self.x = x # input placeholder

        # simple 2-layer network
        W1 = weight_variable([in_dim,50])
        b1 = bias_variable([50])

        W2 = weight_variable([50,out_dim])
        b2 = bias_variable([out_dim])

        h1 = tf.nn.relu(tf.matmul(x,W1) + b1) # hidden layer
        self.y = tf.matmul(h1,W2) + b2 # output layer

        self.var_list = [W1, b1, W2, b2]

        # vanilla single-task loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        self.set_vanilla_loss()

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
