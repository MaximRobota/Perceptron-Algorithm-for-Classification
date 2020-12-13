import numpy as np;

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

tr_inputs = np.array([[0,0,1],
                   [1,1,1],
                   [1,0,1],
                   [0,1,1]])

tr_outputs = np.array([[0,1,1,0]]).T
np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1

int = 200
for i in range(int):
    input_layer = tr_inputs
    outputs = sigmoid( np.dot(input_layer, synaptic_weights) )

    err = tr_outputs - outputs
    adjustments = np.dot( input_layer.T, err * (outputs * (1 - outputs)) )

    synaptic_weights += adjustments

print("synaptic_weights:")
print(synaptic_weights)

print("Result after learning:")
print(outputs)

