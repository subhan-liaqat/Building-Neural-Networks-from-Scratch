import numpy as np

#Dot Product of Single Neuron
inputs = [1, 2, 3, 2.5]
weights = [0.2,0.8, -0.5, 1.0]
bias = 2

#NOTE: weights has to be first, not inputs, cause shape will
#error occur otherwise.
output = np.dot(weights, inputs) + bias
print(output)


##########################################################################


#Dot Product of Neural layer
inputs = [1, 2, 3, 2.5]
weights = [[0.2,0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print(output)
