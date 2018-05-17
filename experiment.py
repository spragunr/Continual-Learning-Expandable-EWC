from model import Model
import torch

BATCH_SIZE = 64


# Create random Tensors to hold inputs and outputs
x = torch.randn(BATCH_SIZE, 784)
y_labels = torch.randn(BATCH_SIZE, 10)

model = Model(x, y_labels)

learning_rate = 1e-4
# parameters() returns an iterator over a list of the trainable model parameters in the same order in
# which they appear in the network when traversed input -> output
# (e.g.
#       [weights b/w input and first hidden layer,
#        bias b/w input and hidden layer 1,
#        ... ,
#        weights between last hidden layer and output,
#        bias b/w hidden layer and output]
# )
optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)

for i in range(500):
    # The output- predictions of classification likelihoods- is produced by running
    # the model on given input data.
    y_pred = model.model(x)

    loss = model.loss(y_pred, y_labels)

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
    accuracy = torch.mean(correct_prediction)

    print(i, loss.item(), accuracy)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()