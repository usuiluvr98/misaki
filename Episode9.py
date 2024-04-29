import numpy as np 

def mcculloch_pitts_neuron(inputs, weights, threshold):
    if len(inputs) != len(weights):
        raise ValueError("Number of inputs must be equal to the number of weights")
    weighted_sum = sum(x * w for x, w in zip(inputs, weights))

    if weighted_sum >= threshold:
        return 1
    else:
        return 0
print("AND Gate")
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
weights = [1, 1]
threshold = 2
for i in range(len(inputs)):    
    input_pattern = inputs[i]
    print("input: ",input_pattern)
    output = mcculloch_pitts_neuron(input_pattern, weights, threshold)
    print("Output:", output)

print("OR Gate")
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
weights = [1, 1]
threshold = 1

for i in range(len(inputs)):    
    input_pattern = inputs[i]
    print("input: ",input_pattern)
    output = mcculloch_pitts_neuron(input_pattern, weights, threshold)
    print("Output:", output)


