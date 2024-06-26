import numpy as np

inputs = np.array([[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]])
outputs = np.array([-1, -1, -1, 1])

weights = np.zeros(inputs.shape[1])
bias = 0

for i in range(len(inputs)):
    input_pattern = inputs[i]
    output_pattern = outputs[i]
    activations = input_pattern
    y = output_pattern
    print("current weight matrix for ",i,"iteration = ",weights)
    weights += np.dot(activations, y)
    bias += y

print("Learned weights:", weights)
